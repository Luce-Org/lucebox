#pragma once

#include "common/pflash_types.h"

#include <cstddef>
#include <string>
#include <vector>

namespace dflash::qwen3 {

enum class PFlashSelectionMode {
    Legacy,
    BudgetOnly,
    CumulativeTopP,
};

enum class PFlashQueryParser {
    SemanticUser,
    ArbitraryTail,
};

enum class PFlashSelectionStop {
    TopPReached,
    BudgetReached,
    CandidatesExhausted,
    InvalidInput,
    MandatoryQueryExceedsBudget,
};

struct PFlashSelectionCandidate {
    size_t ordinal = 0;
    int begin = 0;
    int end = 0;
    double score = 0.0;
    bool mandatory = false;
};

struct PFlashSelectionPolicy {
    int token_budget = 0;
    double top_p = 0.95;
    // Variable-length segments: a candidate that does not fit the remaining
    // budget is skipped instead of ending the fill, so smaller segments
    // ranked below it can still be kept.
    bool skip_oversized = false;
};

struct PFlashSelectionResult {
    bool ok = false;
    std::vector<size_t> ordinals;
    int retained_tokens = 0;
    double retained_mass = 0.0;
    PFlashSelectionStop stop = PFlashSelectionStop::InvalidInput;
    std::string error;
};

bool pflash_chunk_is_structurally_required(
    int begin,
    int end,
    int query_begin,
    int query_end,
    int input_tokens,
    const std::vector<dflash::common::PFlashTokenSpan> &
        required_instruction_spans = {}) noexcept;

bool validate_pflash_instruction_spans(
    const std::vector<dflash::common::PFlashTokenSpan> & spans,
    int input_tokens,
    std::string & error) noexcept;

PFlashSelectionResult select_pflash_candidates(
    const std::vector<PFlashSelectionCandidate> & candidates,
    const PFlashSelectionPolicy & policy,
    PFlashSelectionMode mode);

const char * pflash_selection_mode_name(PFlashSelectionMode mode) noexcept;
const char * pflash_selection_stop_name(PFlashSelectionStop stop) noexcept;
const char * pflash_query_parser_name(PFlashQueryParser parser) noexcept;

// How the context is cut into candidates and how a candidate is scored.
// ``Auto`` resolves at scoring time: probe segments when a segment probe is
// loaded, fixed chunks otherwise; density with probe segments, sum otherwise.
enum class PFlashSegmentation { Auto, Fixed, Probe };
enum class PFlashCandidateScore { Auto, Sum, Density };
// Which scorer ranks the candidates: the block-15 attention-mass head, the
// original all-layer running-max scorer, or both with a split budget (the
// head fills ``split_fraction`` of the budget first, the other scorer the rest).
enum class PFlashScorer { Head, Legacy, Split };

struct PFlashSelectionConfig {
    PFlashSelectionMode mode = PFlashSelectionMode::Legacy;
    PFlashQueryParser query_parser = PFlashQueryParser::SemanticUser;
    int chunk_size = 0;
    int query_tokens = 8;
    double top_p = 0.95;
    PFlashSegmentation segmentation = PFlashSegmentation::Auto;
    PFlashCandidateScore candidate_score = PFlashCandidateScore::Auto;
    PFlashScorer scorer = PFlashScorer::Head;
    double split_fraction = 0.5;
    bool configured = false;
    bool selection_active = false;
};

// Segment probe: cut the context before every token whose boundary score is
// above ``threshold``; ``forced_cuts`` (query start, instruction span edges)
// are always cut; a cut closer than ``min_segment`` tokens to the previous
// accepted cut is dropped unless forced; a span longer than ``max_segment``
// is split at its best-scoring interior token, or evenly when no interior
// token scores above zero. ``split_scores`` (the sub-unit logit when the
// probe artifact carries one) feeds only the oversize interior argmax; when
// empty the unit boundary scores are used. Returns contiguous spans covering
// [0, input_tokens), or an empty vector on invalid input.
std::vector<dflash::common::PFlashTokenSpan> pflash_probe_segments(
    const std::vector<float> & boundary_scores,
    int input_tokens,
    float threshold,
    int min_segment,
    int max_segment,
    const std::vector<int> & forced_cuts,
    const std::vector<float> & split_scores = {});

// Two-scorer selection: ``head`` candidates fill ``head_fraction`` of the
// budget (mandatory candidates first, charged once), then ``other``
// candidates (same spans and ordinals, scored by the other scorer) fill what
// remains, skipping ordinals already selected. Both lists must describe the
// same spans in the same order.
PFlashSelectionResult select_pflash_split(
    const std::vector<PFlashSelectionCandidate> & head,
    const std::vector<PFlashSelectionCandidate> & other,
    const PFlashSelectionPolicy & policy,
    double head_fraction,
    PFlashSelectionMode mode);

const char * pflash_scorer_name(PFlashScorer scorer) noexcept;
const char * pflash_segmentation_name(PFlashSegmentation segmentation) noexcept;
const char * pflash_candidate_score_name(PFlashCandidateScore score) noexcept;

// Presence, rather than validity, gates cache and continuation policy so an
// empty or invalid experiment variable cannot silently fall back to legacy.
bool has_pflash_selection_environment() noexcept;

bool resolve_pflash_selection(
    int input_tokens,
    int legacy_chunk_size,
    PFlashSelectionConfig & out,
    std::string & error);

} // namespace dflash::qwen3
