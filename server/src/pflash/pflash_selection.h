#pragma once

#include "common/pflash_types.h"
#include "pflash_request_probe.h"
#include "pflash_query_candidates.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace luce::pflash {

enum class PFlashSelectionMode {
    Legacy,
    BudgetOnly,
    CumulativeTopP,
    // Rank rule: keep the K highest-scoring optional candidates, the token
    // budget still a hard ceiling -- min(K segments, the budget).
    TopK,
};

enum class PFlashQueryParser {
    SemanticUser,
    ArbitraryTail,
};

enum class PFlashSelectionStop {
    TopPReached,
    TopKReached,
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
    // TopK mode only: how many optional candidates to keep. Must be positive
    // in that mode and is ignored in the others.
    int top_k = 0;
};

struct PFlashSelectionResult {
    bool ok = false;
    std::vector<size_t> ordinals;
    int retained_tokens = 0;
    double retained_mass = 0.0;
    PFlashSelectionStop stop = PFlashSelectionStop::InvalidInput;
    std::string error;
};

// A chunk is kept whatever its score when it overlaps the query window or a
// required instruction span, or -- with ``query_suffix_structural`` -- any
// token after the query window.
bool pflash_chunk_is_structurally_required(
    int begin,
    int end,
    int query_begin,
    int query_end,
    int input_tokens,
    const std::vector<luce::common::PFlashTokenSpan> &
        required_instruction_spans = {},
    bool query_suffix_structural = true) noexcept;

bool validate_pflash_instruction_spans(
    const std::vector<luce::common::PFlashTokenSpan> & spans,
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

struct PFlashRequestProbeTrace;  // pflash_request_probe.h

// Fixed PFlash November Update selection profile (strict budget-only head
// scoring, probe segments, density score, neural query candidates + 512-token
// verbatim pin).
struct PFlashSelectionConfig {
    PFlashSelectionMode mode = PFlashSelectionMode::BudgetOnly;
    PFlashQueryParser query_parser = PFlashQueryParser::ArbitraryTail;
    int chunk_size = 256;
    int query_tokens = 128;
    double top_p = 0.95;
    int top_k = 0;
    PFlashSegmentation segmentation = PFlashSegmentation::Probe;
    PFlashCandidateScore candidate_score = PFlashCandidateScore::Density;
    bool configured = true;
    bool selection_active = true;
    // Per request: the tokens after the query window are candidates scored
    // against it instead of a kept suffix (a chat whose latest user turn is
    // followed by assistant and tool turns). The caller pins whatever of that
    // suffix must stay.
    bool query_suffix_candidates = false;
    // Per request: earlier user questions' scorer windows, most recent
    // first. The head scores the context against each and mixes the masses
    // with the query's at weights 1/2, 1/4, ... (multi-turn chats).
    std::vector<luce::common::PFlashTokenSpan> history_queries;
    // Per request: windows a later request of the same chat will score as
    // history queries but this one does not (the latest user question's
    // counterpart in a tool session, whose prompt ends after tool output).
    // Never scored: the scorer keeps their rows in its session, so the next
    // request can resume instead of scoring from token 0.
    std::vector<luce::common::PFlashTokenSpan> keep_queries;
    // Per request: the tail of the latest user turn, scored as a second
    // query window at full weight next to the prompt-end query.
    luce::common::PFlashTokenSpan turn_query{-1, -1};
    // Request probe + relative query candidates (active whenever the request
    // probe GGUF is loaded).
    bool query_probe = true;
    std::vector<luce::common::PFlashTokenSpan> request_messages;
    std::string request_messages_error;
    std::vector<int> request_human_index;
    std::vector<luce::common::PFlashTokenSpan> request_recent;
    std::vector<int> request_recent_index;
    std::vector<luce::pflash::PFlashRequestMessage> request_messages_all;
    std::string request_messages_all_error;
    // Trace only: what the request probe did on this call (not owned; set
    // by the scorer for the selection it calls).
    const PFlashRequestProbeTrace * request_probe_trace = nullptr;
    PFlashQueryConfig query;
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
std::vector<luce::common::PFlashTokenSpan> pflash_probe_segments(
    const std::vector<float> & boundary_scores,
    int input_tokens,
    float threshold,
    int min_segment,
    int max_segment,
    const std::vector<int> & forced_cuts,
    const std::vector<float> & split_scores = {});

// Line-aligned oversize split (probe GGUF ``segmentprobe.oversize_split`` =
// "line_aligned"). Per token t: ``line_start[t]`` = token t-1's text holds the
// last '\n' before t; ``after_blank[t]`` = the line that '\n' ends holds only
// spaces, tabs or '\r' (or nothing). Derived from the per-token texts alone,
// with no marker strings.
struct PFlashLineFlags {
    std::vector<uint8_t> line_start;
    std::vector<uint8_t> after_blank;
};
PFlashLineFlags pflash_line_flags(const std::vector<std::string> & token_texts);

// As above, but when ``lines`` is given (sized >= input_tokens) an oversize
// cut must fall at a line start inside the same window: among line starts
// that follow a blank line the highest split score, else among all line
// starts the highest split score (earliest on ties); with no line start in
// the window, the cut above. ``lines == nullptr`` is exactly the function
// above.
std::vector<luce::common::PFlashTokenSpan> pflash_probe_segments(
    const std::vector<float> & boundary_scores,
    int input_tokens,
    float threshold,
    int min_segment,
    int max_segment,
    const std::vector<int> & forced_cuts,
    const std::vector<float> & split_scores,
    const PFlashLineFlags * lines);

const char * pflash_segmentation_name(PFlashSegmentation segmentation) noexcept;
const char * pflash_candidate_score_name(PFlashCandidateScore score) noexcept;

bool resolve_pflash_selection(
    int input_tokens,
    int legacy_chunk_size,
    PFlashSelectionConfig & out,
    std::string & error);

// Resolves selection config for a drafter_score_and_compress call:
//   - request_messages == nullptr: standalone contract (caller chunk_size,
//     Auto segmentation/score, request-probe query steps disabled).
//   - request_messages with human spans only: human-only request probe
//     without requiring the full message map.
//   - request_messages with full message map (server / request2): baked
//     November Update server profile.
bool resolve_pflash_drafter_selection(
    int input_tokens,
    int caller_chunk_size,
    const luce::common::PFlashRequestMessages * request_messages,
    PFlashSelectionConfig & out,
    std::string & error);

} // namespace luce::pflash
