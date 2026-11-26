#include "CppUnitTestFramework.hpp"

#include "common/sha256.h"
#include "pflash/pflash_selection.h"
#include "pflash/pflash_compress.h"
#include "pflash/pflash_drafter.h"
#include "pflash/pflash_seg_kcos.h"
#include "pflash/pflash_seg_reranker.h"
#include "scoped_env.h"

#include "ggml-cpu.h"
#if defined(LUCE_HAVE_PFLASH_SEG_KCOS)
#include "ggml-cuda.h"
#endif
#include "gguf.h"
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace luce::pflash;

namespace {

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

TEST_CASE(PFlashSelectionFixture, query_suffix_candidates_are_optional_unless_pinned) {
    // Agent loop: turns after the query are scored context. Only the query
    // window and pinned spans (the generation prompt) stay mandatory.
    constexpr int input_tokens = 200;
    constexpr int query_begin = 20;
    constexpr int query_end = 30;
    const std::vector<luce::common::PFlashTokenSpan> pinned{{190, 200}};
    REQUIRE(pflash_chunk_is_structurally_required(
        0, 32, query_begin, query_end, input_tokens, pinned, false));
    REQUIRE(!pflash_chunk_is_structurally_required(
        32, 64, query_begin, query_end, input_tokens, pinned, false));
    REQUIRE(pflash_chunk_is_structurally_required(
        32, 64, query_begin, query_end, input_tokens, pinned, true));
    REQUIRE(pflash_chunk_is_structurally_required(
        160, 200, query_begin, query_end, input_tokens, pinned, false));
}

TEST_CASE(PFlashSelectionFixture, instruction_overlap_is_mandatory_without_changing_optional_ranking) {
    constexpr int input_tokens = 120000;
    constexpr int query_begin = 119872;
    constexpr int query_end = 120000;
    const std::vector<luce::common::PFlashTokenSpan> instructions{
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
    const std::vector<std::vector<luce::common::PFlashTokenSpan>> invalid{
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
    // No count cap in process (V2-A3): a 200-turn session pins one role
    // header per turn; the IPC message keeps kPFlashMaxInstructionSpans.
    std::vector<luce::common::PFlashTokenSpan> many;
    for (int k = 0; k < 200; ++k) many.push_back({k * 10, k * 10 + 3});
    std::string error;
    REQUIRE(validate_pflash_instruction_spans(many, 2000, error));
    REQUIRE(error.empty());
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

TEST_CASE(PFlashSelectionFixture, resolver_returns_baked_gold_defaults_and_validates_inputs) {
    const auto config = resolve_or_fail(4096, 32);

    REQUIRE(config.configured);
    REQUIRE(config.selection_active);
    REQUIRE(config.mode == PFlashSelectionMode::BudgetOnly);
    REQUIRE(config.query_parser == PFlashQueryParser::ArbitraryTail);
    REQUIRE(config.chunk_size == 256);
    REQUIRE(config.query_tokens == 128);
    REQUIRE(config.segmentation == PFlashSegmentation::Probe);
    REQUIRE(config.candidate_score == PFlashCandidateScore::Density);
    REQUIRE(config.query_probe);
    REQUIRE(config.query.candidates == 1);
    REQUIRE(config.query.pin_tokens == 512);

    PFlashSelectionConfig invalid;
    std::string error;
    REQUIRE(!resolve_pflash_selection(-1, 32, invalid, error));
    REQUIRE(!error.empty());
    error.clear();
    REQUIRE(!resolve_pflash_selection(100, 0, invalid, error));
    REQUIRE(!error.empty());
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

TEST_CASE(PFlashSelectionFixture, line_flags_mark_line_starts_and_blank_lines) {
    // Same cases as the Python port (rules.py line_flags).
    const std::vector<std::string> texts = {"a", "b\n", "  ", "x", "\n\n", "y", " z.\n", "\n", "w"};
    const auto flags = pflash_line_flags(texts);
    const std::vector<uint8_t> starts = {0, 0, 1, 0, 0, 1, 0, 1, 1};
    const std::vector<uint8_t> blanks = {0, 0, 0, 0, 0, 1, 0, 0, 1};
    REQUIRE(flags.line_start == starts);
    REQUIRE(flags.after_blank == blanks);
    // A line of spaces / tabs / CR is blank; a newline in the last token flags nothing.
    const auto ws = pflash_line_flags({"x\n", " \t\r", "\n", "y", "\n"});
    REQUIRE(ws.line_start == std::vector<uint8_t>({0, 1, 0, 1, 0}));
    REQUIRE(ws.after_blank == std::vector<uint8_t>({0, 0, 0, 1, 0}));
}

TEST_CASE(PFlashSelectionFixture, line_aligned_split_moves_oversize_cuts_to_line_starts) {
    std::vector<float> boundary(30, 0.0f);
    std::vector<float> subunit(30, 0.0f);
    subunit[11] = 0.9f;   // best score: mid-line
    subunit[7] = 0.2f;    // a line start
    subunit[9] = 0.1f;    // a line start after a blank line: preferred
    PFlashLineFlags lines;
    lines.line_start.assign(30, 0);
    lines.after_blank.assign(30, 0);
    lines.line_start[7] = 1;
    lines.line_start[9] = 1;
    lines.after_blank[9] = 1;
    const auto plain = pflash_probe_segments(boundary, 30, 0.9f, 2, 12, {}, subunit);
    REQUIRE(plain[0].end == 11);
    const auto aligned = pflash_probe_segments(boundary, 30, 0.9f, 2, 12, {}, subunit, &lines);
    REQUIRE(aligned[0].begin == 0 && aligned[0].end == 9);
    // Without a blank-line start: the highest-scoring line start.
    lines.after_blank[9] = 0;
    const auto any_line = pflash_probe_segments(boundary, 30, 0.9f, 2, 12, {}, subunit, &lines);
    REQUIRE(any_line[0].end == 7);
    // Ties: the earliest line start.
    subunit[9] = 0.2f;
    const auto tie = pflash_probe_segments(boundary, 30, 0.9f, 2, 12, {}, subunit, &lines);
    REQUIRE(tie[0].end == 7);
    // No line start in the window: today's cut.
    PFlashLineFlags none;
    none.line_start.assign(30, 0);
    none.after_blank.assign(30, 0);
    REQUIRE(pflash_probe_segments(boundary, 30, 0.9f, 2, 12, {}, subunit, &none) == plain);
    // Line flags shorter than the input: invalid.
    PFlashLineFlags short_flags;
    short_flags.line_start.assign(5, 0);
    short_flags.after_blank.assign(5, 0);
    REQUIRE(pflash_probe_segments(boundary, 30, 0.9f, 2, 12, {}, subunit, &short_flags).empty());
    // Threshold cuts and the floor are untouched; spans stay contiguous and capped.
    boundary[20] = 0.95f;
    const auto cut = pflash_probe_segments(boundary, 30, 0.9f, 2, 12, {}, subunit, &lines);
    int prev = 0;
    bool has20 = false;
    for (const auto & span : cut) {
        REQUIRE(span.begin == prev);
        REQUIRE(span.end - span.begin <= 12);
        prev = span.end;
        has20 = has20 || span.begin == 20;
    }
    REQUIRE(prev == 30 && has20);
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

namespace {

// Six equal-length optional candidates, scores descending with the ordinal.
std::vector<PFlashSelectionCandidate> ranked_candidates() {
    std::vector<PFlashSelectionCandidate> candidates;
    for (size_t index = 0; index < 6; ++index) {
        const int begin = (int) index * 100;
        candidates.push_back(
            candidate(index, begin, begin + 100, 6.0 - (double) index));
    }
    return candidates;
}

} // namespace

TEST_CASE(PFlashSelectionFixture, top_k_keeps_the_k_highest_scoring_optional_candidates) {
    const auto candidates = ranked_candidates();
    // Budget far above the six candidates: K alone decides.
    const auto result = select_pflash_candidates(
        candidates, PFlashSelectionPolicy{100000, 0.95, false, 2},
        PFlashSelectionMode::TopK);

    REQUIRE(result.ok);
    REQUIRE(result.stop == PFlashSelectionStop::TopKReached);
    REQUIRE(result.retained_tokens == 200);
    require_ordinals(result, {0, 1});
}

TEST_CASE(PFlashSelectionFixture, top_k_above_the_candidate_count_falls_back_to_budget_behaviour) {
    const auto candidates = ranked_candidates();
    // K larger than the candidate list: every candidate fits, so the run ends
    // exactly where budget_only would.
    const auto roomy = select_pflash_candidates(
        candidates, PFlashSelectionPolicy{100000, 0.95, false, 50},
        PFlashSelectionMode::TopK);
    const auto budget_only = select_pflash_candidates(
        candidates, PFlashSelectionPolicy{100000, 0.95, false},
        PFlashSelectionMode::BudgetOnly);

    REQUIRE(roomy.ok);
    REQUIRE(roomy.stop == PFlashSelectionStop::CandidatesExhausted);
    REQUIRE(roomy.stop == budget_only.stop);
    REQUIRE(roomy.retained_tokens == budget_only.retained_tokens);
    require_ordinals(roomy, {0, 1, 2, 3, 4, 5});

    // The same K against a budget that bites: the budget stops the fill.
    const auto tight = select_pflash_candidates(
        candidates, PFlashSelectionPolicy{250, 0.95, false, 50},
        PFlashSelectionMode::TopK);
    REQUIRE(tight.ok);
    REQUIRE(tight.stop == PFlashSelectionStop::BudgetReached);
    REQUIRE(tight.retained_tokens == 200);
    require_ordinals(tight, {0, 1});
}

TEST_CASE(PFlashSelectionFixture, top_k_skips_an_oversized_candidate_and_keeps_ranking) {
    // Variable-length segments: the 300-token second-ranked candidate does not
    // fit, so the fill continues below it instead of ending, exactly as
    // budget_only does, and K counts only the candidates actually kept.
    const std::vector<PFlashSelectionCandidate> candidates{
        candidate(0, 0, 100, 10.0),
        candidate(1, 100, 400, 9.0),
        candidate(2, 400, 500, 8.0),
        candidate(3, 500, 600, 7.0),
    };
    const auto result = select_pflash_candidates(
        candidates, PFlashSelectionPolicy{250, 0.95, true, 3},
        PFlashSelectionMode::TopK);

    REQUIRE(result.ok);
    REQUIRE(result.stop == PFlashSelectionStop::BudgetReached);
    REQUIRE(result.retained_tokens == 200);
    require_ordinals(result, {0, 2});
}

TEST_CASE(PFlashSelectionFixture, top_k_keeps_mandatory_candidates_outside_the_rank) {
    // Ordinal 3 is mandatory and scores lowest: it is kept and charged, and K
    // still buys one optional candidate on top of it.
    std::vector<PFlashSelectionCandidate> candidates{
        candidate(0, 0, 100, 5.0),
        candidate(1, 100, 200, 4.0),
        candidate(2, 200, 300, 3.0),
        candidate(3, 300, 400, 0.0, true),
    };
    const auto result = select_pflash_candidates(
        candidates, PFlashSelectionPolicy{100000, 0.95, false, 1},
        PFlashSelectionMode::TopK);

    REQUIRE(result.ok);
    REQUIRE(result.stop == PFlashSelectionStop::TopKReached);
    REQUIRE(result.retained_tokens == 200);
    require_ordinals(result, {0, 3});

    // A mandatory span that cannot fit still fails closed under top_k.
    const auto overflow = select_pflash_candidates(
        candidates, PFlashSelectionPolicy{50, 0.95, false, 1},
        PFlashSelectionMode::TopK);
    REQUIRE(!overflow.ok);
    REQUIRE(overflow.stop == PFlashSelectionStop::MandatoryQueryExceedsBudget);
}

TEST_CASE(PFlashSelectionFixture, top_k_requires_a_positive_k) {
    const auto candidates = ranked_candidates();
    REQUIRE(!select_pflash_candidates(
        candidates, PFlashSelectionPolicy{100000, 0.95, false, 0},
        PFlashSelectionMode::TopK).ok);
    REQUIRE(!select_pflash_candidates(
        candidates, PFlashSelectionPolicy{100000, 0.95, false, -1},
        PFlashSelectionMode::TopK).ok);
    // The other modes ignore the field.
    REQUIRE(select_pflash_candidates(
        candidates, PFlashSelectionPolicy{100000, 0.95, false, 0},
        PFlashSelectionMode::BudgetOnly).ok);
    REQUIRE(std::string(pflash_selection_mode_name(PFlashSelectionMode::TopK)) == "top_k");
    REQUIRE(std::string(pflash_selection_stop_name(PFlashSelectionStop::TopKReached)) ==
            "top_k_reached");
}

// ═════════════════════════════════════════════════
// Score validation
// ═════════════════════════════════════════════════

TEST_CASE(PFlashSelectionFixture, score_validation_counts_nan_and_inf) {
    const float values[]{
        0.0f,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        1.0f,
    };
    CHECK(luce::common::count_nonfinite_scores(values, 5) == 3);
    CHECK(luce::common::count_nonfinite_scores(values, 1) == 0);
}

// ═════════════════════════════════════════════════
// Segment reranker: kcos features, mass lift, adaptive budget
// ═════════════════════════════════════════════════

TEST_CASE(PFlashSelectionFixture, adjacent_segment_key_cosine_cpu_matches_analytic_values) {
    // 4 tokens, k_dim = 2:
    //   seg 0 [0,1): (1, 0)
    //   seg 1 [1,2): (1, 1) -> cos with seg 0 = 1/sqrt(2)
    //   seg 2 [2,3): (0, 1) -> cos with seg 1 = 1/sqrt(2)
    //   seg 3 [3,4): (0,-1) -> cos with seg 2 = -1 -> clamped to 0
    const std::vector<float> keys{
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, -1.0f,
    };
    const std::vector<luce::common::PFlashTokenSpan> segs{{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    std::vector<float> kcos;
    luce::common::pflash_adjacent_segment_key_cosine_cpu(keys.data(), 4, 2, segs, kcos);
    REQUIRE(kcos.size() == 4u);
    REQUIRE(kcos[0] == 0.0f);
    REQUIRE(std::fabs(kcos[1] - 0.70710678f) < 1e-5f);
    REQUIRE(std::fabs(kcos[2] - 0.70710678f) < 1e-5f);
    REQUIRE(kcos[3] == 0.0f);
}

TEST_CASE(PFlashSelectionFixture, adjacent_segment_key_cosine_cpu_computes_katt_and_zero_mass_fallback) {
    // 4 tokens, k_dim = 2, 2 segments [0,2) and [2,4):
    //   seg 0: t0=(1,0) mass=0.0, t1=(0,1) mass=1e-6 -> k_bar=(1/sqrt2, 1/sqrt2), k_att=(0, 1)
    //   seg 1: t2=(1,0) mass=0.0, t3=(1,0) mass=0.0  -> zero mass -> k_att falls back to k_bar=(1, 0)
    const std::vector<float> keys{
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 0.0f,
    };
    const std::vector<float> mass{0.0f, 1e-6f, 0.0f, 0.0f};
    const std::vector<luce::common::PFlashTokenSpan> segs{{0, 2}, {2, 4}};
    std::vector<float> kcos, kbar, katt;
    luce::common::pflash_adjacent_segment_key_cosine_cpu(
        keys.data(), 4, 2, segs, kcos, &kbar, mass.data(), &katt);
    REQUIRE(kbar.size() == 4u);
    REQUIRE(katt.size() == 4u);
    REQUIRE(std::fabs(kbar[0] - 0.70710678f) < 1e-5f);
    REQUIRE(std::fabs(kbar[1] - 0.70710678f) < 1e-5f);
    REQUIRE(std::fabs(katt[0] - 0.0f) < 1e-5f);
    REQUIRE(std::fabs(katt[1] - 1.0f) < 1e-5f);
    REQUIRE(std::fabs(katt[2] - 1.0f) < 1e-5f);
    REQUIRE(std::fabs(katt[3] - 0.0f) < 1e-5f);

    // Optional GPU vs CPU parity check on realistic 512d keys (D=256, Hk=2) when PFLASH_TEST_GPU_KCOS=1.
    const char * gpu_env = std::getenv("PFLASH_TEST_GPU_KCOS");
    if (gpu_env && std::string(gpu_env) == "1") {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        REQUIRE(dev != nullptr);
        ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
        REQUIRE(gpu != nullptr);
        const int D = 256, Hk = 2, n_tok = 32, k_dim = D * Hk;
        std::vector<float> h_keys((size_t) n_tok * (size_t) k_dim);
        for (int t = 0; t < n_tok; ++t) {
            for (int d = 0; d < k_dim; ++d) {
                h_keys[(size_t) t * (size_t) k_dim + (size_t) d] =
                    std::sin(0.13f * (float) (t + 1) + 0.07f * (float) d);
            }
        }
        std::vector<float> h_mass((size_t) n_tok, 0.0f);
        for (int t = 0; t < 12; ++t) h_mass[(size_t) t] = (t == 5) ? 0.02f : 0.0005f;
        for (int t = 12; t < 22; ++t) h_mass[(size_t) t] = (t == 17) ? 1e-7f : 1e-9f; // tiny mass scaling
        // seg 2 [22,32): all zero mass -> must fall back to k_bar!
        const std::vector<luce::common::PFlashTokenSpan> gsegs{{0, 12}, {12, 22}, {22, 32}};

        std::vector<float> kcos_cpu, kbar_cpu, katt_cpu;
        luce::common::pflash_adjacent_segment_key_cosine_cpu(
            h_keys.data(), n_tok, k_dim, gsegs, kcos_cpu, &kbar_cpu, h_mass.data(), &katt_cpu);

        // Also verify NaN and large-magnitude (1e20) parity: GPU must write explicit zeros
        // (not NaN) on invalid rows and use double-precision norm so 1e20 does not overflow.
        std::vector<float> h_edge((size_t) 6 * (size_t) k_dim, 0.0f);
        for (int d = 0; d < k_dim; ++d) {
            h_edge[(size_t) 0 * k_dim + d] = std::numeric_limits<float>::quiet_NaN();
            h_edge[(size_t) 1 * k_dim + d] = 1.0f;
            h_edge[(size_t) 2 * k_dim + d] = (d == 0) ? 1e20f : 0.0f;
            h_edge[(size_t) 3 * k_dim + d] = (d == 0) ? 1e20f : 0.0f;
            h_edge[(size_t) 4 * k_dim + d] = (d == 1) ? 1e20f : 0.0f;
            h_edge[(size_t) 5 * k_dim + d] = (d == 1) ? 1e20f : 0.0f;
        }
        std::vector<float> h_edge_mass(6, 1.0f);
        const std::vector<luce::common::PFlashTokenSpan> esegs{{0, 2}, {2, 4}, {4, 6}};
        std::vector<float> ekcos_cpu, ekbar_cpu, ekatt_cpu;
        luce::common::pflash_adjacent_segment_key_cosine_cpu(
            h_edge.data(), 6, k_dim, esegs, ekcos_cpu, &ekbar_cpu, h_edge_mass.data(), &ekatt_cpu);

        ggml_init_params ip{};
        ip.mem_size = 8 * ggml_tensor_overhead() + 4096;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        REQUIRE(ctx != nullptr);
#if defined(LUCE_HAVE_PFLASH_SEG_KCOS)
        REQUIRE(ggml_backend_is_cuda(gpu));
#endif
        ggml_tensor * keys_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hk, n_tok);
        ggml_tensor * edge_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, Hk, 6);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        REQUIRE(buf != nullptr);
        ggml_backend_tensor_set(keys_t, h_keys.data(), 0, h_keys.size() * sizeof(float));
        ggml_backend_tensor_set(edge_t, h_edge.data(), 0, h_edge.size() * sizeof(float));

        std::vector<float> kcos_gpu, kbar_gpu, katt_gpu;
        REQUIRE(luce::common::pflash_adjacent_segment_key_cosine(
            gpu, keys_t, gsegs, kcos_gpu, &kbar_gpu, h_mass.data(), &katt_gpu));
        REQUIRE(kcos_gpu.size() == kcos_cpu.size());
        REQUIRE(kbar_gpu.size() == kbar_cpu.size());
        REQUIRE(katt_gpu.size() == katt_cpu.size());
        for (size_t i = 0; i < kcos_cpu.size(); ++i) REQUIRE(std::fabs(kcos_gpu[i] - kcos_cpu[i]) < 1e-5f);
        for (size_t i = 0; i < kbar_cpu.size(); ++i) REQUIRE(std::fabs(kbar_gpu[i] - kbar_cpu[i]) < 1e-5f);
        for (size_t i = 0; i < katt_cpu.size(); ++i) REQUIRE(std::fabs(katt_gpu[i] - katt_cpu[i]) < 1e-5f);

        std::vector<float> ekcos_gpu, ekbar_gpu, ekatt_gpu;
        REQUIRE(luce::common::pflash_adjacent_segment_key_cosine(
            gpu, edge_t, esegs, ekcos_gpu, &ekbar_gpu, h_edge_mass.data(), &ekatt_gpu));
        for (size_t i = 0; i < ekbar_gpu.size(); ++i) {
            REQUIRE(std::isfinite(ekbar_gpu[i]));
            REQUIRE(std::isfinite(ekatt_gpu[i]));
            REQUIRE(std::fabs(ekbar_gpu[i] - ekbar_cpu[i]) < 1e-5f);
            REQUIRE(std::fabs(ekatt_gpu[i] - ekatt_cpu[i]) < 1e-5f);
        }
        REQUIRE(std::fabs(ekbar_gpu[(size_t) 1 * k_dim + 0] - 1.0f) < 1e-5f);
        REQUIRE(std::fabs(ekbar_gpu[(size_t) 2 * k_dim + 1] - 1.0f) < 1e-5f);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(gpu);
    }
}

TEST_CASE(PFlashSelectionFixture, droppable_mass_lift_excludes_token0_sink_and_mandatory) {
    // 4 segments of 4 tokens each ([0,4), [4,8), [8,12), [12,16) mandatory).
    // Token 0 holds a giant 0.80 sink; remaining optional tokens have uniform 0.01 except
    // seg 2 [8,12) which has 0.04 per token.
    std::vector<float> scores(16, 0.01f);
    scores[0] = 0.80f;
    for (int t = 8; t < 12; ++t) scores[t] = 0.04f;
    std::vector<PFlashSelectionCandidate> cands{
        candidate(0,  0,  4, 0.2075, false),
        candidate(1,  4,  8, 0.01,   false),
        candidate(2,  8, 12, 0.04,   false),
        candidate(3, 12, 16, 0.01,   true),
    };
    // Selecting only seg 2 (4 of 11 non-sink optional tokens; mass = 0.16 of 0.03 + 0.04 + 0.16 = 0.23):
    // lift = (0.16 / 0.23) / (4 / 11) = 1.91304...
    const double lift = luce::common::pflash_droppable_mass_lift(cands, scores, {0, 0, 1, 1});
    REQUIRE(std::fabs(lift - ((0.16 / 0.23) / (4.0 / 11.0))) < 1e-6);
    // Selecting seg 0 (contains sink token 0): sink excluded so mass = 0.03, len = 3:
    // lift = (0.03 / 0.23) / (3 / 11) = 0.47826... (NOT inflated by token 0!).
    const double lift_sink = luce::common::pflash_droppable_mass_lift(cands, scores, {1, 0, 0, 1});
    REQUIRE(std::fabs(lift_sink - ((0.03 / 0.23) / (3.0 / 11.0))) < 1e-6);
}

TEST_CASE(PFlashSelectionFixture, select_pflash_chunks_reranker_separates_rank_score_from_stage1_lift) {
    // 40 tokens = 4 optional segments of 8 tokens + 1 mandatory query segment [32,40).
    // Reranker overrides selection order and .rank_score while .lift keeps raw calibrated Stage-1 mass.
    std::vector<int32_t> ids(40);
    for (int i = 0; i < 40; ++i) ids[(size_t) i] = i;
    std::vector<float> mass(40, 0.001f);
    for (int t = 8; t < 16; ++t) mass[(size_t) t] = 0.08f;   // seg 1 peak
    for (int t = 16; t < 24; ++t) mass[(size_t) t] = 0.02f;  // seg 2 medium
    std::vector<luce::common::PFlashTokenSpan> segs{{0, 8}, {8, 16}, {16, 24}, {24, 32}, {32, 40}};

    std::vector<float> rerank{5.0f, 4.0f, -1.0f, -2.0f, 0.0f};
    PFlashSelectionConfig rcfg;
    rcfg.mode = PFlashSelectionMode::BudgetOnly;
    rcfg.selection_active = true;
    luce::common::pflash_clear_kept_spans();
    const auto rout = luce::common::select_pflash_chunks(
        ids, mass, /*keep_ratio=*/0.60f, /*n_lookahead=*/8, /*score_query_end=*/40,
        /*pool_kernel=*/1, rcfg, {}, /*direct_mass=*/true, /*write_trace=*/false,
        &segs, /*density=*/true, &rerank);
    REQUIRE(rout.size() == 24u);
    const auto & rlifts = luce::common::pflash_last_candidate_lifts();
    REQUIRE(rlifts.size() == 5u);
    REQUIRE(rlifts[0].lift < rlifts[2].lift);
    REQUIRE(std::fabs(rlifts[0].lift - 0.001 * 40.0) < 1e-6);
    REQUIRE(rlifts[0].rank_score > rlifts[2].rank_score);
    REQUIRE(std::fabs(rlifts[0].rank_score - std::exp(5.0) * 40.0) < 1e-3);
    const auto & rkept = luce::common::pflash_last_kept_spans();
    REQUIRE(!rkept.empty() && rkept[0].begin == 0 && rkept[0].end == 16);

    // Reranker scores above 20 must preserve ordering rather than collapsing at exp(20).
    std::vector<float> high_rerank{23.0f, 1.0f, 70.0f, -2.0f, 0.0f};
    luce::common::pflash_clear_kept_spans();
    const auto hout = luce::common::select_pflash_chunks(
        ids, mass, /*keep_ratio=*/0.40f, /*n_lookahead=*/8, /*score_query_end=*/40,
        /*pool_kernel=*/1, rcfg, {}, /*direct_mass=*/true, /*write_trace=*/false,
        &segs, /*density=*/true, &high_rerank);
    REQUIRE(hout.size() == 16u);
    const auto & hkept = luce::common::pflash_last_kept_spans();
    REQUIRE(hkept.size() == 2u);
    REQUIRE(hkept[0].begin == 16 && hkept[0].end == 24);
    const auto & hlifts = luce::common::pflash_last_candidate_lifts();
    REQUIRE(hlifts[2].rank_score > hlifts[0].rank_score);

    // Sum scoring: a reranked score is already per token and keeps its scale; the mandatory
    // query segment's sum is still divided by its length.
    luce::common::pflash_clear_kept_spans();
    luce::common::select_pflash_chunks(
        ids, mass, /*keep_ratio=*/0.60f, /*n_lookahead=*/8, /*score_query_end=*/40,
        /*pool_kernel=*/1, rcfg, {}, /*direct_mass=*/true, /*write_trace=*/false,
        &segs, /*density=*/false, &rerank);
    const auto & slifts = luce::common::pflash_last_candidate_lifts();
    REQUIRE(slifts.size() == 5u);
    REQUIRE(std::fabs(slifts[0].lift - 0.001 * 40.0) < 1e-6);
    REQUIRE(std::fabs(slifts[0].rank_score - std::exp(5.0) * 40.0) < 1e-3);
    REQUIRE(std::fabs(slifts[4].rank_score - 0.001 * 40.0) < 1e-6);
}

TEST_CASE(PFlashSelectionFixture, select_pflash_chunks_trace_records_stage1_scores_when_reranked) {
    namespace fs = std::filesystem;
    const fs::path trace_path = fs::temp_directory_path() / "luce_pflash_stage1_trace_test.jsonl";
    std::error_code ec;
    fs::remove(trace_path, ec);

    luce_test::ScopedEnvVar trace_env{"PFLASH_TRACE_PATH", trace_path.string().c_str()};
    std::vector<int32_t> ids(40);
    for (int i = 0; i < 40; ++i) ids[(size_t) i] = i;
    std::vector<float> mass(40, 0.001f);
    for (int t = 8; t < 16; ++t) mass[(size_t) t] = 0.08f;
    std::vector<luce::common::PFlashTokenSpan> segs{{0, 8}, {8, 16}, {16, 24}, {24, 32}, {32, 40}};
    std::vector<float> rerank{5.0f, 4.0f, -1.0f, -2.0f, 0.0f};
    PFlashSelectionConfig cfg;
    cfg.mode = PFlashSelectionMode::BudgetOnly;
    cfg.selection_active = true;

    // 1. With reranker: trace records both reranked `chunk_scores` and raw `stage1_chunk_scores`.
    const auto out_rr = luce::common::select_pflash_chunks(
        ids, mass, 0.60f, 8, 40, 1, cfg, {}, true, true, &segs, true, &rerank);
    REQUIRE(!out_rr.empty());

    // 2. Without reranker: `stage1_chunk_scores` is omitted.
    const auto out_s1 = luce::common::select_pflash_chunks(
        ids, mass, 0.60f, 8, 40, 1, cfg, {}, true, true, &segs, true, nullptr);
    REQUIRE(!out_s1.empty());

    std::ifstream ifs(trace_path);
    REQUIRE(ifs.good());
    std::string line1, line2;
    REQUIRE(static_cast<bool>(std::getline(ifs, line1)));
    REQUIRE(static_cast<bool>(std::getline(ifs, line2)));
    const auto j1 = nlohmann::json::parse(line1);
    const auto j2 = nlohmann::json::parse(line2);
    REQUIRE(j1.contains("stage1_chunk_scores"));
    REQUIRE(j1.at("stage1_chunk_scores").size() == 5u);
    REQUIRE(std::fabs(j1.at("stage1_chunk_scores")[0].get<double>() - 0.001) < 1e-6);
    REQUIRE(std::fabs(j1.at("stage1_chunk_scores")[1].get<double>() - 0.08) < 1e-6);
    REQUIRE(std::fabs(j1.at("chunk_scores")[0].get<double>() - std::exp(5.0)) < 1e-3);
    REQUIRE(!j2.contains("stage1_chunk_scores"));

    fs::remove(trace_path, ec);
}

TEST_CASE(PFlashSelectionFixture, reranker_adaptive_budget_scales_with_spread_and_bypasses_flat) {
    // 1. Sharp 32K prompt (even with chat header [0, 3) mandatory so first optional starts at token 3!):
    //    4 relevant segments (1600 tok) at +4.5 above 76 background segments -> eff < 0.42 -> keep_ratio=0 (base curve).
    std::vector<luce::common::PFlashTokenSpan> segs;
    std::vector<float> scores;
    std::vector<uint8_t> mand;
    segs.push_back({0, 3});  // mandatory chat template prefix `<|im_start|>user\n`
    scores.push_back(0.0f);
    mand.push_back(1);
    int pos = 3;
    for (int i = 0; i < 80; ++i) {
        segs.push_back({pos, pos + 400});
        pos += 400;
        // First optional segment (i == 0, begin == 3) is a BOS-adjacent sink (+10.0).
        float s = (i == 0) ? -1.0f : (i >= 20 && i < 24 ? -6.5f : -11.0f);
        scores.push_back(s);
        mand.push_back(0);
    }
    const auto sharp = luce::common::pflash_reranker_adaptive_budget(segs, scores, mand);
    REQUIRE(sharp.valid);
    REQUIRE(!sharp.bypass_to_full);
    REQUIRE(sharp.eff < 0.25f);
    REQUIRE(sharp.keep_ratio == 0.0f);  // a sharp prompt keeps the base curve

    // 2. Diffuse / global prompt: flat/mild scores across body, even with a giant +10 sink on first optional seg (begin=3).
    std::vector<float> flat(81, -11.0f);
    flat[0] = 0.0f;
    flat[1] = -1.0f;  // +10 BOS-adjacent sink at first optional segment (begin == 3) must be skipped!
    for (int i = 2; i <= 11; ++i) flat[(size_t) i] = -9.5f;  // mild +1.5 spread across 10 segments
    const auto diffuse = luce::common::pflash_reranker_adaptive_budget(segs, flat, mand);
    REQUIRE(diffuse.valid);
    REQUIRE(diffuse.eff >= 0.58f);
    REQUIRE(diffuse.bypass_to_full);
    REQUIRE(diffuse.keep_ratio == 1.0f);

    // 3. Moderately broad / multi-file prompt (0.42 <= eff < 0.58): widens monotonically without bypassing.
    std::vector<float> broad(81, -11.0f);
    broad[0] = 0.0f;
    broad[1] = -1.0f;
    for (int i = 2; i <= 21; ++i) broad[(size_t) i] = -9.0f;  // 20 segments at +2.0 above median -> eff ~ 0.48
    const auto widened = luce::common::pflash_reranker_adaptive_budget(segs, broad, mand);
    REQUIRE(widened.valid);
    REQUIRE(!widened.bypass_to_full);
    REQUIRE(widened.eff >= 0.42f && widened.eff < 0.58f);
    REQUIRE(std::fabs(widened.keep_ratio - widened.eff) < 1e-6f);

    // 4. Nucleus Stage-1 mass floor (90% mass, capped at 0.58):
    //    4a. Sharp Stage-1 distribution (92% of mass in the top-4 body segments [21..24] plus BOS seg [1] = 5/80 = 6.25%)
    //        -> nucleus_ratio ~ 0.0625 (well below base curve, so sharp prompts stay fast).
    std::vector<double> sharp_masses(81, 0.001);
    sharp_masses[0] = 0.0;
    for (int i = 21; i <= 24; ++i) sharp_masses[(size_t) i] = 0.23;  // 0.92 / (0.92 + 0.076) = 92.4%
    const auto nuc_sharp = luce::common::pflash_reranker_adaptive_budget(segs, scores, mand, &sharp_masses);
    REQUIRE(nuc_sharp.valid);
    REQUIRE(!nuc_sharp.bypass_to_full);
    REQUIRE(nuc_sharp.nucleus_ratio <= 0.07f);
    REQUIRE(nuc_sharp.keep_ratio <= 0.07f);

    //    4b. Fat-tailed Stage-1 distribution (top 4 reranked segments only hold 55% of Stage-1 mass;
    //        need 36/80 = 45% optional segments to reach 90% mass) while eff < 0.42
    //        -> nucleus_ratio widens keep_ratio to ~0.45 without bypassing.
    std::vector<double> fat_masses(81, 0.002);  // 44 tail segs * 0.002 = 0.088
    fat_masses[0] = 0.0;
    for (int i = 1; i <= 4; ++i) fat_masses[(size_t) i] = 0.1375;   // 0.55
    for (int i = 5; i <= 36; ++i) fat_masses[(size_t) i] = 0.0113;  // 0.3616 -> top 36 hold 91.2%
    std::vector<float> fat_scores(81, -11.0f);
    fat_scores[0] = 0.0f;
    for (int i = 1; i <= 4; ++i) fat_scores[(size_t) i] = -6.5f;   // sharp reranked spike -> eff < 0.42
    for (int i = 5; i <= 36; ++i) fat_scores[(size_t) i] = -10.5f; // ranked right after top 4
    const auto nuc_fat = luce::common::pflash_reranker_adaptive_budget(segs, fat_scores, mand, &fat_masses);
    REQUIRE(nuc_fat.valid);
    REQUIRE(!nuc_fat.bypass_to_full);
    REQUIRE(nuc_fat.eff < 0.42f);
    REQUIRE(nuc_fat.nucleus_ratio >= 0.42f && nuc_fat.nucleus_ratio <= 0.47f);
    REQUIRE(std::fabs(nuc_fat.keep_ratio - nuc_fat.nucleus_ratio) < 1e-6f);

    //    4c. Uniform Stage-1 mass (needs 72/80 = 90% tokens) with sharp reranked scores (eff < 0.58)
    //        -> nucleus_ratio cleanly capped at 0.58 without triggering full bypass.
    std::vector<double> unif_masses(81, 0.01);
    unif_masses[0] = 0.0;
    const auto nuc_cap = luce::common::pflash_reranker_adaptive_budget(segs, scores, mand, &unif_masses);
    REQUIRE(nuc_cap.valid);
    REQUIRE(!nuc_cap.bypass_to_full);
    REQUIRE(std::fabs(nuc_cap.nucleus_ratio - 0.58f) < 1e-6f);
    REQUIRE(std::fabs(nuc_cap.keep_ratio - 0.58f) < 1e-6f);
}

TEST_CASE(PFlashSelectionFixture, seg_reranker_refuses_another_head_or_feature_mode) {
    const auto dir = std::filesystem::temp_directory_path() / "luce_pflash_seg_reranker_contract_test";
    std::filesystem::create_directories(dir);
    const auto write = [&](const char * name, const char * mode) {
        gguf_context * g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "pflash_seg_reranker");
        gguf_set_val_str(g, "reranker.schema", "pflash_seg_reranker_v1");
        gguf_set_val_str(g, "reranker.feature_mode", mode);
        gguf_set_val_str(g, "reranker.runtime_gguf_sha256", "drafter");
        gguf_set_val_str(g, "reranker.scoring_head_sha256", "head-a");
        const std::string path = (dir / name).string();
        gguf_write_to_file(g, path.c_str(), /*only_meta=*/ false);
        gguf_free(g);
        return path;
    };
    const std::string kbar = write("kbar.gguf", "kbar");
    const std::string other_mode = write("other.gguf", "kbar_q");
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    luce::common::PFlashSegRerankerState st;
    std::string err;

    CHECK(!luce::common::load_pflash_seg_reranker(kbar, cpu, "drafter", "head-b", st, err));
    CHECK(err.find("scoring head sha256 mismatch") != std::string::npos);
    CHECK(!luce::common::load_pflash_seg_reranker(other_mode, cpu, "drafter", "head-a", st, err));
    CHECK(err.find("feature mode kbar_q") != std::string::npos);
    // The head it names passes the contract; this stub then stops at its missing dimensions.
    CHECK(!luce::common::load_pflash_seg_reranker(kbar, cpu, "drafter", "head-a", st, err));
    CHECK(err.find("dimension metadata missing") != std::string::npos);

    // Malformed 2D bias tensor [d_model, 2] where 1D [d_model] is expected must fail cleanly, not abort.
    {
        gguf_context * g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "pflash_seg_reranker");
        gguf_set_val_str(g, "reranker.schema", "pflash_seg_reranker_v1");
        gguf_set_val_str(g, "reranker.feature_mode", "kbar");
        gguf_set_val_str(g, "reranker.runtime_gguf_sha256", "drafter");
        gguf_set_val_str(g, "reranker.scoring_head_sha256", "head-a");
        gguf_set_val_u32(g, "reranker.in_dim", 4);
        gguf_set_val_u32(g, "reranker.n_scalars", 12);
        gguf_set_val_u32(g, "reranker.d_model", 4);
        gguf_set_val_u32(g, "reranker.n_heads", 1);
        gguf_set_val_u32(g, "reranker.n_layers", 1);
        gguf_set_val_u32(g, "reranker.d_ff", 4);
        gguf_set_val_f32(g, "reranker.rope_theta", 10000.0f);
        gguf_set_val_f32(g, "reranker.norm_eps", 1e-6f);
        ggml_init_params gp{16 * ggml_tensor_overhead() + 4096, nullptr, false};
        ggml_context * gctx = ggml_init(gp);
        ggml_tensor * w = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 4, 4);
        ggml_set_name(w, "reranker.k_in.weight");
        ggml_tensor * bad_b = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 4, 2);
        ggml_set_name(bad_b, "reranker.k_in.bias");
        gguf_add_tensor(g, w);
        gguf_add_tensor(g, bad_b);
        const std::string bad_path = (dir / "bad_shape.gguf").string();
        gguf_write_to_file(g, bad_path.c_str(), /*only_meta=*/false);
        gguf_free(g);
        ggml_free(gctx);
        CHECK(!luce::common::load_pflash_seg_reranker(bad_path, cpu, "drafter", "head-a", st, err));
        CHECK(err.find("reranker.k_in.bias") != std::string::npos);
    }

    // Malformed metadata: huge n_layers must fail before allocating layer vectors, and
    // non-finite / non-positive rope_theta or norm_eps must be rejected cleanly.
    const auto write_meta = [&](const char * name, uint32_t n_layers, float rope_theta, float norm_eps) {
        gguf_context * g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "pflash_seg_reranker");
        gguf_set_val_str(g, "reranker.schema", "pflash_seg_reranker_v1");
        gguf_set_val_str(g, "reranker.feature_mode", "kbar");
        gguf_set_val_str(g, "reranker.runtime_gguf_sha256", "drafter");
        gguf_set_val_str(g, "reranker.scoring_head_sha256", "head-a");
        gguf_set_val_u32(g, "reranker.in_dim", 4);
        gguf_set_val_u32(g, "reranker.n_scalars", 12);
        gguf_set_val_u32(g, "reranker.d_model", 4);
        gguf_set_val_u32(g, "reranker.n_heads", 1);
        gguf_set_val_u32(g, "reranker.n_layers", n_layers);
        gguf_set_val_u32(g, "reranker.d_ff", 4);
        gguf_set_val_f32(g, "reranker.rope_theta", rope_theta);
        gguf_set_val_f32(g, "reranker.norm_eps", norm_eps);
        const std::string path = (dir / name).string();
        gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
        gguf_free(g);
        return path;
    };
    {
        const std::string huge_layers = write_meta("huge_layers.gguf", 1000000000u, 10000.0f, 1e-6f);
        CHECK(!luce::common::load_pflash_seg_reranker(huge_layers, cpu, "drafter", "head-a", st, err));
        CHECK(err.find("invalid dimensions") != std::string::npos);
        CHECK(!luce::common::validate_pflash_seg_reranker_gguf(huge_layers, "drafter", "head-a", 4, err));

        const std::string bad_eps = write_meta("bad_eps.gguf", 1u, 10000.0f, 0.0f);
        CHECK(!luce::common::load_pflash_seg_reranker(bad_eps, cpu, "drafter", "head-a", st, err));
        CHECK(err.find("rope_theta or norm_eps") != std::string::npos);

        const std::string nan_eps = write_meta("nan_eps.gguf", 1u, 10000.0f, std::numeric_limits<float>::quiet_NaN());
        CHECK(!luce::common::load_pflash_seg_reranker(nan_eps, cpu, "drafter", "head-a", st, err));
        CHECK(err.find("rope_theta or norm_eps") != std::string::npos);

        const std::string bad_rope = write_meta("bad_rope.gguf", 1u, -10.0f, 1e-6f);
        CHECK(!luce::common::load_pflash_seg_reranker(bad_rope, cpu, "drafter", "head-a", st, err));
        CHECK(err.find("rope_theta or norm_eps") != std::string::npos);

        const std::string inf_rope = write_meta("inf_rope.gguf", 1u, std::numeric_limits<float>::infinity(), 1e-6f);
        CHECK(!luce::common::load_pflash_seg_reranker(inf_rope, cpu, "drafter", "head-a", st, err));
        CHECK(err.find("rope_theta or norm_eps") != std::string::npos);
    }

    ggml_backend_free(cpu);
    std::filesystem::remove_all(dir);
}

TEST_CASE(PFlashSelectionFixture, standalone_drafter_selection_preserves_caller_chunk_and_skips_query_step) {
    // Standalone call (request_messages == nullptr): N=128, keep_ratio=0.5, chunk_size=32 must
    // honor caller chunk_size=32 (not baked 256) and disable request-probe query steps.
    PFlashSelectionConfig standalone;
    std::string err;
    REQUIRE(resolve_pflash_drafter_selection(128, 32, nullptr, standalone, err));
    CHECK(standalone.chunk_size == 32);
    CHECK(standalone.segmentation == PFlashSegmentation::Auto);
    CHECK(standalone.candidate_score == PFlashCandidateScore::Auto);
    CHECK(!standalone.query_probe);
    CHECK(standalone.query.candidates == 0);
    CHECK(standalone.query.pin_tokens == 0);

    // Selecting on 128 tokens at keep_ratio=0.5 with chunk_size=32 keeps 64 tokens cleanly.
    std::vector<int32_t> ids(128, 7);
    std::vector<float> scores(128, 0.01f);
    scores[40] = 0.9f;
    const auto kept = luce::common::select_pflash_chunks(
        ids, scores, /*keep_ratio=*/0.5f, /*n_lookahead=*/8, /*score_query_end=*/128,
        /*pool_kernel=*/1, standalone, {}, /*direct_mass=*/true, /*write_trace=*/false);
    REQUIRE(kept.size() == 64u);

    // Human-only request_messages (pflash_daemon `request`): keeps request probe active,
    // honors caller chunk_size, disables full-map query candidates/pin — even with zero human spans.
    luce::common::PFlashRequestMessages human_only;
    human_only.human_only = true;
    PFlashSelectionConfig human_empty_cfg;
    REQUIRE(resolve_pflash_drafter_selection(128, 64, &human_only, human_empty_cfg, err));
    CHECK(human_empty_cfg.chunk_size == 64);
    CHECK(human_empty_cfg.query_probe);
    CHECK(human_empty_cfg.query.candidates == 0);
    CHECK(human_empty_cfg.query.pin_tokens == 0);

    human_only.human.push_back({10, 40});
    PFlashSelectionConfig human_cfg;
    REQUIRE(resolve_pflash_drafter_selection(128, 64, &human_only, human_cfg, err));
    CHECK(human_cfg.chunk_size == 64);
    CHECK(human_cfg.query_probe);
    CHECK(human_cfg.query.candidates == 0);
    CHECK(human_cfg.query.pin_tokens == 0);

    // Full message map (server / pflash_daemon `request2`): keeps the baked November profile.
    luce::common::PFlashRequestMessages full_msgs;
    full_msgs.all_roles = {"user"};
    full_msgs.all_spans = {{0, 128}};
    PFlashSelectionConfig server_cfg;
    REQUIRE(resolve_pflash_drafter_selection(65536, 32, &full_msgs, server_cfg, err));
    CHECK(server_cfg.chunk_size == 256);
    CHECK(server_cfg.segmentation == PFlashSegmentation::Probe);
    CHECK(server_cfg.candidate_score == PFlashCandidateScore::Density);
    CHECK(server_cfg.query_probe);
    CHECK(server_cfg.query.candidates == 1);
    CHECK(server_cfg.query.pin_tokens == 512);
}

TEST_CASE(PFlashSelectionFixture, startup_gguf_contract_validator_rejects_stubs) {
    const auto dir = std::filesystem::temp_directory_path() / "luce_pflash_validate_stubs";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto stub = [&](const char * name) {
        const std::string p = (dir / name).string();
        std::ofstream(p) << "not-a-gguf";
        return p;
    };
    const std::string d = stub("drafter.gguf");
    const std::string h = stub("seg-scoring-head.gguf");
    const std::string s = stub("segment-probe.gguf");
    const std::string r = stub("request-probe.gguf");
    const std::string rr = stub("seg-reranker.gguf");
    std::string err;
    CHECK(!luce::common::validate_pflash_component_ggufs(d, h, s, r, rr, err));
    CHECK(!err.empty());

    // Wrong-typed general.architecture (u32 instead of string) and wrong-typed general.alignment
    // (string instead of u32) must fail cleanly, never abort.
    {
        gguf_context * g = gguf_init_empty();
        gguf_set_val_u32(g, "general.architecture", 42u);
        const std::string bad_drafter = (dir / "bad_arch_type.gguf").string();
        gguf_write_to_file(g, bad_drafter.c_str(), /*only_meta=*/false);
        gguf_free(g);
        CHECK(!luce::common::validate_pflash_component_ggufs(bad_drafter, h, s, r, rr, err));
        CHECK(!err.empty());
    }
    {
        // Craft a raw GGUF v3 header with general.alignment stored as GGUF_TYPE_STRING (8)
        // since gguf_set_val_str() itself refuses to emit one.
        const std::string bad_align = (dir / "bad_align_type.gguf").string();
        std::ofstream out(bad_align, std::ios::binary);
        const char magic[4] = {'G', 'G', 'U', 'F'};
        const uint32_t ver = 3;
        const uint64_t n_tensors = 0, n_kv = 1;
        const std::string key = "general.alignment";
        const uint64_t klen = key.size();
        const uint32_t ktype = 8; // GGUF_TYPE_STRING
        const std::string val = "32";
        const uint64_t vlen = val.size();
        out.write(magic, 4);
        out.write(reinterpret_cast<const char *>(&ver), sizeof(ver));
        out.write(reinterpret_cast<const char *>(&n_tensors), sizeof(n_tensors));
        out.write(reinterpret_cast<const char *>(&n_kv), sizeof(n_kv));
        out.write(reinterpret_cast<const char *>(&klen), sizeof(klen));
        out.write(key.data(), (std::streamsize) key.size());
        out.write(reinterpret_cast<const char *>(&ktype), sizeof(ktype));
        out.write(reinterpret_cast<const char *>(&vlen), sizeof(vlen));
        out.write(val.data(), (std::streamsize) val.size());
        out.close();
        CHECK(!luce::common::validate_pflash_component_ggufs(bad_align, h, s, r, rr, err));
        CHECK(!err.empty());
    }
    // A drafter GGUF with valid Qwen3.5 metadata but truncated tensor payload must also be rejected.
    {
        gguf_context * g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "qwen35");
        gguf_set_val_u32(g, "qwen35.embedding_length", 16u);
        gguf_set_val_u32(g, "qwen35.attention.head_count", 2u);
        gguf_set_val_u32(g, "qwen35.attention.head_count_kv", 1u);
        gguf_set_val_u32(g, "qwen35.attention.key_length", 8u);
        gguf_set_val_u32(g, "qwen35.block_count", 16u);
        gguf_set_val_u32(g, "qwen35.full_attention_interval", 4u);
        ggml_init_params gp{4 * ggml_tensor_overhead() + 4096, nullptr, false};
        ggml_context * gctx = ggml_init(gp);
        ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 32);
        ggml_set_name(t, "token_embd.weight");
        gguf_add_tensor(g, t);
        const std::string trunc_drafter = (dir / "trunc_drafter.gguf").string();
        gguf_write_to_file(g, trunc_drafter.c_str(), /*only_meta=*/false);
        gguf_free(g);
        ggml_free(gctx);
        const auto full_sz = std::filesystem::file_size(trunc_drafter);
        std::filesystem::resize_file(trunc_drafter, full_sz - 4);
        CHECK(!luce::common::validate_pflash_component_ggufs(trunc_drafter, h, s, r, rr, err));
        CHECK(err.find("truncated data section") != std::string::npos);
    }
    // Cold-start validation must ignore a stale same-size `.sha256` sidecar beside the drafter
    // and hash the actual drafter bytes instead.
    {
        const std::string stale_drafter = (dir / "stale_sidecar_drafter.gguf").string();
        const std::string stale_head = (dir / "stale_sidecar_head.gguf").string();
        const std::string fake_sha(64, 'a');
        {
            gguf_context * g = gguf_init_empty();
            gguf_set_val_str(g, "general.architecture", "qwen35");
            gguf_set_val_u32(g, "qwen35.embedding_length", 16u);
            gguf_set_val_u32(g, "qwen35.attention.head_count", 2u);
            gguf_set_val_u32(g, "qwen35.attention.head_count_kv", 1u);
            gguf_set_val_u32(g, "qwen35.attention.key_length", 8u);
            gguf_set_val_u32(g, "qwen35.block_count", 16u);
            gguf_set_val_u32(g, "qwen35.full_attention_interval", 4u);
            ggml_init_params gp{4 * ggml_tensor_overhead() + 4096, nullptr, false};
            ggml_context * gctx = ggml_init(gp);
            ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 32);
            ggml_set_name(t, "token_embd.weight");
            gguf_add_tensor(g, t);
            gguf_write_to_file(g, stale_drafter.c_str(), /*only_meta=*/false);
            gguf_free(g);
            ggml_free(gctx);
        }
        const auto drafter_sz = std::filesystem::file_size(stale_drafter);
        {
            std::ofstream sidecar(stale_drafter + ".sha256");
            sidecar << fake_sha << "  stale_sidecar_drafter.gguf\n# size=" << drafter_sz << "\n";
        }
        {
            gguf_context * g = gguf_init_empty();
            gguf_set_val_str(g, "general.architecture", "pflash_scoring_head");
            gguf_set_val_str(g, "scoringhead.schema", "qwen3_5_0_8b_nope_qk_mass_v1");
            gguf_set_val_str(g, "scoringhead.base_model", "Qwen/Qwen3.5-0.8B");
            gguf_set_val_str(g, "scoringhead.runtime_gguf_sha256", fake_sha.c_str());
            gguf_set_val_str(g, "scoringhead.feature_tap", "post_block14_residual_before_block15");
            ggml_init_params gp{8 * ggml_tensor_overhead() + 8192, nullptr, false};
            ggml_context * gctx = ggml_init(gp);
            ggml_tensor * wq = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 16, 16);
            ggml_set_name(wq, "scoringhead.attn_q.weight");
            ggml_tensor * wk = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 16, 8);
            ggml_set_name(wk, "scoringhead.attn_k.weight");
            gguf_add_tensor(g, wq);
            gguf_add_tensor(g, wk);
            gguf_write_to_file(g, stale_head.c_str(), /*only_meta=*/false);
            gguf_free(g);
            ggml_free(gctx);
        }
        CHECK(!luce::common::validate_pflash_component_ggufs(stale_drafter, stale_head, s, r, rr, err));
        CHECK(err.find("scoring head metadata does not match") != std::string::npos);
    }
    std::filesystem::remove_all(dir);
}

// A 1-layer, d_model=4 `pflash_seg_reranker_v1` GGUF bound to drafter "drafter" and head "head-a".
struct TinyRerankerFile {
    ggml_type matmul_type = GGML_TYPE_F32;   // 2D matmul weights
    bool small_matmul_f32 = false;           // keep s_in and the [D,1] heads F32, as the published file does
    ggml_type vector_type = GGML_TYPE_F32;   // 1D norms and biases
    float score_head_bias = 0.05f;
};

static std::string write_tiny_reranker(const std::filesystem::path & path, const TinyRerankerFile & f) {
    const int I = 4, S = 12, D = 4, F = 4;
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "pflash_seg_reranker");
    gguf_set_val_str(g, "reranker.schema", "pflash_seg_reranker_v1");
    gguf_set_val_str(g, "reranker.feature_mode", "kbar");
    gguf_set_val_str(g, "reranker.runtime_gguf_sha256", "drafter");
    gguf_set_val_str(g, "reranker.scoring_head_sha256", "head-a");
    gguf_set_val_u32(g, "reranker.in_dim", I);
    gguf_set_val_u32(g, "reranker.n_scalars", S);
    gguf_set_val_u32(g, "reranker.d_model", D);
    gguf_set_val_u32(g, "reranker.n_heads", 1);
    gguf_set_val_u32(g, "reranker.n_layers", 1);
    gguf_set_val_u32(g, "reranker.d_ff", F);
    gguf_set_val_f32(g, "reranker.rope_theta", 10000.0f);
    gguf_set_val_f32(g, "reranker.norm_eps", 1e-6f);
    ggml_init_params gp{64 * ggml_tensor_overhead() + 65536, nullptr, false};
    ggml_context * gctx = ggml_init(gp);
    // Each tensor holds seed plus a small ripple, so the parity check sees distinct weights.
    const auto add = [&](const char * name, ggml_type type, int64_t ne0, int64_t ne1, float seed) {
        ggml_tensor * t = ne1 == 0 ? ggml_new_tensor_1d(gctx, type, ne0) : ggml_new_tensor_2d(gctx, type, ne0, ne1);
        ggml_set_name(t, name);
        const int64_t n = ggml_nelements(t);
        std::vector<float> vals((size_t) n);
        for (int64_t i = 0; i < n; ++i) vals[(size_t) i] = seed + 0.03f * (float) ((i * 7) % 11 - 5);
        if (type == GGML_TYPE_F32) {
            std::memcpy(t->data, vals.data(), (size_t) n * sizeof(float));
        } else {
            ggml_fp32_to_fp16_row(vals.data(), (ggml_fp16_t *) t->data, n);
        }
        gguf_add_tensor(g, t);
    };
    const auto add1 = [&](const char * name, int64_t ne0, float seed) { add(name, f.vector_type, ne0, 0, seed); };
    const auto add2 = [&](const char * name, int64_t ne0, int64_t ne1, float seed, bool small = false) {
        add(name, (small && f.small_matmul_f32) ? GGML_TYPE_F32 : f.matmul_type, ne0, ne1, seed);
    };
    add2("reranker.k_in.weight", I, D, 0.12f);
    add1("reranker.k_in.bias", D, 0.01f);
    add2("reranker.q_in.weight", I, D, -0.09f);
    add1("reranker.q_in.bias", D, -0.02f);
    add2("reranker.s_in.weight", S, D, 0.08f, true);
    add1("reranker.s_in.bias", D, 0.03f);
    add2("reranker.int_in.weight", D, D, 0.11f);
    add1("reranker.in_norm.weight", D, 1.0f);
    add1("reranker.out_norm.weight", D, 1.0f);
    add2("reranker.score_head.weight", D, 1, 0.15f, true);
    add1("reranker.score_head.bias", 1, f.score_head_bias);
    add2("reranker.pool_gate.weight", D, 1, -0.07f, true);
    add1("reranker.pool_gate.bias", 1, 0.02f);
    add2("reranker.budget_head.weight", D, 1, 0.13f, true);
    add1("reranker.budget_head.bias", 1, -0.04f);
    add1("reranker.blocks.0.norm1.weight", D, 1.0f);
    add2("reranker.blocks.0.q_proj.weight", D, D, 0.09f);
    add2("reranker.blocks.0.k_proj.weight", D, D, -0.08f);
    add2("reranker.blocks.0.v_proj.weight", D, D, 0.07f);
    add2("reranker.blocks.0.o_proj.weight", D, D, -0.06f);
    add1("reranker.blocks.0.norm2.weight", D, 1.0f);
    add2("reranker.blocks.0.gate_proj.weight", D, F, 0.10f);
    add2("reranker.blocks.0.up_proj.weight", D, F, -0.05f);
    add2("reranker.blocks.0.down_proj.weight", F, D, 0.06f);
    gguf_write_to_file(g, path.string().c_str(), /*only_meta=*/false);
    gguf_free(g);
    ggml_free(gctx);
    return path.string();
}

TEST_CASE(PFlashSelectionFixture, seg_reranker_rejects_truncated_file_and_nonfinite_output) {
    const auto dir = std::filesystem::temp_directory_path() / "luce_pflash_seg_reranker_run_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string valid_path = write_tiny_reranker(dir / "tiny_rr.gguf", {});
    TinyRerankerFile nan_head;
    nan_head.score_head_bias = std::numeric_limits<float>::quiet_NaN();
    const std::string nan_head_path = write_tiny_reranker(dir / "nan_head_rr.gguf", nan_head);

    // 1. Truncated payload (even just 1 byte of trailing alignment padding) must be rejected.
    const std::string trunc_path = (dir / "trunc_rr.gguf").string();
    {
        std::filesystem::copy_file(valid_path, trunc_path);
        const auto sz = std::filesystem::file_size(trunc_path);
        std::filesystem::resize_file(trunc_path, sz - 1);
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    luce::common::PFlashSegRerankerState st;
    std::string err;
    CHECK(!luce::common::validate_pflash_seg_reranker_gguf(trunc_path, "drafter", "head-a", 4, err));
    CHECK(err.find("truncated") != std::string::npos);
    CHECK(!luce::common::load_pflash_seg_reranker(trunc_path, cpu, "drafter", "head-a", st, err));
    CHECK(err.find("truncated") != std::string::npos);

    // 2. Valid tiny reranker loads and runs cleanly on finite inputs; a NaN at the final
    //    score head bypasses softmax assertions (safe in Debug builds) and trips the runner guard.
    REQUIRE(luce::common::validate_pflash_seg_reranker_gguf(valid_path, "drafter", "head-a", 4, err));
    REQUIRE(luce::common::load_pflash_seg_reranker(valid_path, cpu, "drafter", "head-a", st, err));
    CHECK(!st.loaded_file_identity.empty());
    std::vector<float> seg_vec(2 * 4, 0.25f);
    std::vector<float> q_vec(4, 0.25f);
    std::vector<float> scalars(2 * 12, 0.1f);
    std::vector<float> log_dens{-2.0f, -3.0f};
    std::vector<float> out_scores;
    float m_star = 0.0f;
    REQUIRE(luce::common::pflash_seg_reranker_run(
        st, cpu, 2, seg_vec.data(), q_vec.data(), scalars.data(), log_dens.data(),
        out_scores, nullptr, &m_star, &err));
    REQUIRE(out_scores.size() == 2u);

    REQUIRE(luce::common::load_pflash_seg_reranker(nan_head_path, cpu, "drafter", "head-a", st, err));
    CHECK(!luce::common::pflash_seg_reranker_run(
        st, cpu, 2, seg_vec.data(), q_vec.data(), scalars.data(), log_dens.data(),
        out_scores, nullptr, &m_star, &err));
    CHECK(out_scores.empty());
    CHECK(err.find("non-finite") != std::string::npos);

    luce::common::free_pflash_seg_reranker(st);
    ggml_backend_free(cpu);
    std::filesystem::remove_all(dir);
}

TEST_CASE(PFlashSelectionFixture, seg_reranker_fp16_weights_load_and_parity) {
    const auto dir = std::filesystem::temp_directory_path() / "luce_pflash_seg_reranker_fp16_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);


    TinyRerankerFile f16_all, f16_mix, bad_1d_f16;
    f16_all.matmul_type = f16_mix.matmul_type = bad_1d_f16.matmul_type = GGML_TYPE_F16;
    f16_mix.small_matmul_f32 = true;
    bad_1d_f16.vector_type = GGML_TYPE_F16;
    const std::string f32_path = write_tiny_reranker(dir / "rr_f32.gguf", {});
    const std::string f16_all_path = write_tiny_reranker(dir / "rr_f16_all.gguf", f16_all);
    const std::string f16_mix_path = write_tiny_reranker(dir / "rr_f16_mix.gguf", f16_mix);
    const std::string bad_1d_f16_path = write_tiny_reranker(dir / "rr_bad_1d_f16.gguf", bad_1d_f16);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    luce::common::PFlashSegRerankerState st_f32, st_f16;
    std::string err;

    REQUIRE(luce::common::validate_pflash_seg_reranker_gguf(f32_path, "drafter", "head-a", 4, err));
    REQUIRE(luce::common::load_pflash_seg_reranker(f32_path, cpu, "drafter", "head-a", st_f32, err));

    const std::vector<float> seg_vec{0.2f, -0.3f, 0.5f, 0.1f, -0.4f, 0.25f, 0.15f, -0.2f, 0.35f, 0.1f, -0.15f, 0.45f};
    const std::vector<float> q_vec{0.3f, -0.1f, 0.2f, 0.4f};
    std::vector<float> scalars(3 * 12, 0.0f);
    for (size_t i = 0; i < scalars.size(); ++i) scalars[i] = -0.5f + 0.07f * (float) (i % 13);
    const std::vector<float> log_dens{-2.1f, -3.4f, -1.5f};

    std::vector<float> scores_f32, delta_f32;
    float m_f32 = 0.0f;
    REQUIRE(luce::common::pflash_seg_reranker_run(
        st_f32, cpu, 3, seg_vec.data(), q_vec.data(), scalars.data(), log_dens.data(),
        scores_f32, &delta_f32, &m_f32, &err));

    for (const auto & [path, expect_sin_f32] : std::vector<std::pair<std::string, bool>>{
             {f16_all_path, false}, {f16_mix_path, true}}) {
        REQUIRE(luce::common::validate_pflash_seg_reranker_gguf(path, "drafter", "head-a", 4, err));
        REQUIRE(luce::common::load_pflash_seg_reranker(path, cpu, "drafter", "head-a", st_f16, err));
        CHECK(st_f16.k_in_w->type == GGML_TYPE_F16);
        CHECK(st_f16.s_in_w->type == (expect_sin_f32 ? GGML_TYPE_F32 : GGML_TYPE_F16));

        std::vector<float> scores_f16, delta_f16;
        float m_f16 = 0.0f;
        REQUIRE(luce::common::pflash_seg_reranker_run(
            st_f16, cpu, 3, seg_vec.data(), q_vec.data(), scalars.data(), log_dens.data(),
            scores_f16, &delta_f16, &m_f16, &err));
        REQUIRE(scores_f16.size() == 3u);
        for (size_t i = 0; i < 3; ++i) {
            CHECK(std::fabs(delta_f16[i] - delta_f32[i]) < 5e-3f);
            CHECK(std::fabs(scores_f16[i] - scores_f32[i]) < 5e-3f);
        }
        CHECK(std::fabs(m_f16 - m_f32) < 5e-3f);
    }

    // 1D norms/biases stored as FP16 must be rejected (ggml_add/mul require F32).
    CHECK(!luce::common::validate_pflash_seg_reranker_gguf(bad_1d_f16_path, "drafter", "head-a", 4, err));
    CHECK(err.find("tensor mismatch") != std::string::npos);

    luce::common::free_pflash_seg_reranker(st_f32);
    luce::common::free_pflash_seg_reranker(st_f16);
    ggml_backend_free(cpu);
    std::filesystem::remove_all(dir);
}

TEST_CASE(PFlashSelectionFixture, scoring_head_fp16_weights_pass_validation) {
    const auto dir = std::filesystem::temp_directory_path() / "luce_pflash_scoring_head_fp16_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const std::string drafter = (dir / "drafter.gguf").string();
    {
        gguf_context * g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "qwen35");
        gguf_set_val_u32(g, "qwen35.embedding_length", 16u);
        gguf_set_val_u32(g, "qwen35.attention.head_count", 2u);
        gguf_set_val_u32(g, "qwen35.attention.head_count_kv", 1u);
        gguf_set_val_u32(g, "qwen35.attention.key_length", 8u);
        gguf_set_val_u32(g, "qwen35.block_count", 16u);
        gguf_set_val_u32(g, "qwen35.full_attention_interval", 4u);
        ggml_init_params gp{4 * ggml_tensor_overhead() + 4096, nullptr, false};
        ggml_context * gctx = ggml_init(gp);
        ggml_tensor * t = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 32);
        ggml_set_name(t, "token_embd.weight");
        gguf_add_tensor(g, t);
        gguf_write_to_file(g, drafter.c_str(), /*only_meta=*/false);
        gguf_free(g);
        ggml_free(gctx);
    }
    std::string drafter_sha;
    {
        std::ifstream ifs(drafter, std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(ifs)), {});
        luce::common::Sha256Ctx c;
        luce::common::sha256_init(c);
        luce::common::sha256_update(c, bytes.data(), bytes.size());
        drafter_sha = luce::common::sha256_final(c);
    }

    const auto write_head = [&](const char * name, ggml_type wtype) {
        const std::string path = (dir / name).string();
        gguf_context * g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "pflash_scoring_head");
        gguf_set_val_str(g, "scoringhead.schema", "qwen3_5_0_8b_nope_qk_mass_v1");
        gguf_set_val_str(g, "scoringhead.base_model", "Qwen/Qwen3.5-0.8B");
        gguf_set_val_str(g, "scoringhead.runtime_gguf_sha256", drafter_sha.c_str());
        gguf_set_val_str(g, "scoringhead.feature_tap", "post_block14_residual_before_block15");
        ggml_init_params gp{8 * ggml_tensor_overhead() + 8192, nullptr, false};
        ggml_context * gctx = ggml_init(gp);
        ggml_tensor * wq = ggml_new_tensor_2d(gctx, wtype, 16, 16);
        ggml_set_name(wq, "scoringhead.attn_q.weight");
        ggml_tensor * wk = ggml_new_tensor_2d(gctx, wtype, 16, 8);
        ggml_set_name(wk, "scoringhead.attn_k.weight");
        gguf_add_tensor(g, wq);
        gguf_add_tensor(g, wk);
        gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
        gguf_free(g);
        ggml_free(gctx);
        return path;
    };

    const std::string head_f16 = write_head("head_f16.gguf", GGML_TYPE_F16);
    const std::string head_bf16 = write_head("head_bf16.gguf", GGML_TYPE_BF16);
    const std::string stub = (dir / "stub.gguf").string();
    std::ofstream(stub) << "not-a-gguf";

    // FP16 scoring head passes head validation and proceeds to segment-probe validation.
    std::string err;
    CHECK(!luce::common::validate_pflash_component_ggufs(drafter, head_f16, stub, stub, stub, err));
    CHECK(err.find("segment probe GGUF could not be opened") != std::string::npos);

    // Truncated FP16 scoring head is rejected before moving past head validation.
    const std::string trunc_f16 = (dir / "head_f16_trunc.gguf").string();
    std::filesystem::copy_file(head_f16, trunc_f16);
    std::filesystem::resize_file(trunc_f16, std::filesystem::file_size(trunc_f16) - 1);
    CHECK(!luce::common::validate_pflash_component_ggufs(drafter, trunc_f16, stub, stub, stub, err));
    CHECK(err.find("scoring head GGUF truncated data section") != std::string::npos);

    // Unsupported weight type (BF16) is rejected cleanly.
    CHECK(!luce::common::validate_pflash_component_ggufs(drafter, head_bf16, stub, stub, stub, err));
    CHECK(err.find("scoring head tensor contract mismatch") != std::string::npos);

    std::filesystem::remove_all(dir);
}

TEST_CASE(PFlashSelectionFixture, seg_reranker_scalars_and_pytorch_parity) {
    // 1. Verify scalar builder produces finite, well-formed 12-feature rows + exact log_density
    std::vector<float> tok_mass(30, 0.01f);
    tok_mass[12] = 0.25f;
    tok_mass[13] = 0.15f;
    std::vector<luce::common::PFlashTokenSpan> segs{{0, 10}, {10, 20}, {20, 30}};
    std::vector<float> cont(30, 0.1f);
    cont[10] = 0.9f;
    std::vector<float> kcos{0.0f, 0.2f, 0.85f};
    std::vector<float> scalars, log_dens;
    luce::common::pflash_reranker_build_scalars(30, tok_mass, segs, &cont, &kcos, scalars, log_dens);
    REQUIRE(scalars.size() == 3 * 12);
    REQUIRE(log_dens.size() == 3);
    for (float v : scalars) REQUIRE(std::isfinite(v));
    REQUIRE(log_dens[1] > log_dens[0]);
    REQUIRE(std::fabs(scalars[1 * 12 + 8] - 0.9f) < 1e-6f);   // cont_left[1]
    REQUIRE(std::fabs(scalars[0 * 12 + 9] - 0.9f) < 1e-6f);   // cont_right[0]
    REQUIRE(std::fabs(scalars[1 * 12 + 10] - 0.2f) < 1e-6f);  // kcos_left[1]
    REQUIRE(std::fabs(scalars[1 * 12 + 11] - 0.85f) < 1e-6f); // kcos_right[1]

    // 2. If PFLASH_RERANKER_PARITY_GGUF and PFLASH_RERANKER_PARITY_JSON are provided,
    //    verify exact ggml <-> PyTorch numerical parity (<1e-4 max abs diff across delta, scores, m_star).
    const char * gguf_env = std::getenv("PFLASH_RERANKER_PARITY_GGUF");
    const char * json_env = std::getenv("PFLASH_RERANKER_PARITY_JSON");
    if (gguf_env && *gguf_env && json_env && *json_env) {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        REQUIRE(cpu != nullptr);
        luce::common::PFlashSegRerankerState st;
        std::string err;
        const bool loaded = luce::common::load_pflash_seg_reranker(gguf_env, cpu, "", "", st, err);
        REQUIRE(loaded);
        std::ifstream ifs(json_env);
        REQUIRE(ifs.good());
        nlohmann::json root;
        ifs >> root;
        const auto & cases = root.is_object() && root.contains("cases") ? root.at("cases") : root;
        REQUIRE(cases.is_array());
        REQUIRE(!cases.empty());
        for (const auto & c : cases) {
            const int n_seg = c.at("n_seg").get<int>();
            std::vector<float> seg_vec, q_vec, sc_vec, ld_vec;
            if (!c.at("seg_vec").empty() && c.at("seg_vec")[0].is_array()) {
                for (const auto & row : c.at("seg_vec")) {
                    for (const auto & v : row) seg_vec.push_back(v.get<float>());
                }
            } else {
                for (const auto & v : c.at("seg_vec")) seg_vec.push_back(v.get<float>());
            }
            for (const auto & v : c.at("q_vec")) q_vec.push_back(v.get<float>());
            if (!c.at("scalars").empty() && c.at("scalars")[0].is_array()) {
                for (const auto & row : c.at("scalars")) {
                    for (const auto & v : row) sc_vec.push_back(v.get<float>());
                }
            } else {
                for (const auto & v : c.at("scalars")) sc_vec.push_back(v.get<float>());
            }
            for (const auto & v : c.at("log_density")) ld_vec.push_back(v.get<float>());
            std::vector<float> out_scores, out_delta;
            float out_m_star = 0.0f;
            std::string run_err;
            const bool ok = luce::common::pflash_seg_reranker_run(
                st, cpu, n_seg, seg_vec.data(), q_vec.data(), sc_vec.data(), ld_vec.data(),
                out_scores, &out_delta, &out_m_star, &run_err);
            REQUIRE(ok);
            const auto & exp_delta = c.contains("expected_delta") ? c.at("expected_delta") : c.at("delta");
            const auto & exp_scores = c.contains("expected_scores") ? c.at("expected_scores") : c.at("scores");
            const float exp_m = c.contains("expected_m_star")
                ? c.at("expected_m_star").get<float>()
                : c.at("pred_m").get<float>();
            float max_d = 0.0f;
            for (int i = 0; i < n_seg; ++i) {
                const float dd = std::fabs(out_delta[(size_t) i] - exp_delta[(size_t) i].get<float>());
                const float ds = std::fabs(out_scores[(size_t) i] - exp_scores[(size_t) i].get<float>());
                if (dd > max_d) max_d = dd;
                if (ds > max_d) max_d = ds;
            }
            const float dm = std::fabs(out_m_star - exp_m);
            REQUIRE(max_d < 1e-4f);
            REQUIRE(dm < 1e-4f);
        }
        luce::common::free_pflash_seg_reranker(st);
        ggml_backend_free(cpu);
    }
}
