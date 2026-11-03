#pragma once

#include "common/pflash_types.h"
#include "pflash_ident_query.h"
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
// Which scorer ranks the candidates: the block-15 attention-mass head, the
// original all-layer running-max scorer, or both with a split budget (the
// head fills ``split_fraction`` of the budget first, the other scorer the rest).
enum class PFlashScorer { Head, Legacy, Split };
// How the kept spans are formed: whole segments in score order (today), or
// fold -- anchors completed to their enclosing code or text block
// (pflash_fold.h), routed by prompt type.
enum class PFlashSelectStrategy { Segments, Fold };

struct PFlashRequestProbeTrace;  // pflash_request_probe.h

struct PFlashSelectionConfig {
    PFlashSelectionMode mode = PFlashSelectionMode::Legacy;
    // Chat-first default: the scorer query is the tail of the latest user
    // turn. latest_user stays selectable for benchmark experiments.
    PFlashQueryParser query_parser = PFlashQueryParser::ArbitraryTail;
    int chunk_size = 0;
    int query_tokens = 8;
    double top_p = 0.95;
    int top_k = 0;
    PFlashSegmentation segmentation = PFlashSegmentation::Auto;
    PFlashCandidateScore candidate_score = PFlashCandidateScore::Auto;
    PFlashScorer scorer = PFlashScorer::Head;
    double split_fraction = 0.5;
    PFlashSelectStrategy strategy = PFlashSelectStrategy::Segments;
    // Fold only: completions kept, block size above which a completion keeps
    // the block's first ``fold_head`` tokens plus the anchor.
    int fold_k = 20;
    int fold_cap = 1500;
    int fold_head = 150;
    // Fold only: blank-line paragraphs as the last-resort text blocks.
    bool fold_paragraphs = true;
    bool configured = false;
    bool selection_active = false;
    // Per request, never from the environment: the tokens after the query
    // window are candidates scored against it instead of a kept suffix (a
    // chat whose latest user turn is followed by assistant and tool turns).
    // The caller pins whatever of that suffix must stay.
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
    // query window at full weight next to the prompt-end query. The last
    // token reads the whole request; the user's own tokens match literal
    // strings (an identifier, a function description) the last token does
    // not carry.
    luce::common::PFlashTokenSpan turn_query{-1, -1};
    // PFLASH_SELECT_QUERY_HEAD=1 (latest_user parser): the question may sit
    // before pasted material. The server scores the user's leading prose
    // block (capped at ``query_head_tokens``) and the trailing prose block
    // (capped at ``query_tail_tokens``) as query windows; a block over its
    // cap contributes its first and last cap/2 tokens.
    bool query_head = false;
    int query_head_tokens = 256;
    int query_tail_tokens = 256;
    // Per request: those extra windows, scored at full weight next to the
    // query window and kept verbatim.
    std::vector<luce::common::PFlashTokenSpan> extra_queries;
    // PFLASH_SELECT_QUERY_PROBE=1 (strict selection, head scorer): the
    // scorer query is the request text of the latest human user messages,
    // found by the probe's request row (pflash_request_probe.h); the
    // content spans of those messages, latest first, come per request.
    bool query_probe = false;
    std::vector<luce::common::PFlashTokenSpan> request_messages;
    // Why the human messages' spans are unusable ("" when they are not).
    std::string request_messages_error;
    // Candidate-rule inputs: the human messages' indices and every
    // non-system messages (spans and indices), read when the GGUF's candidate
    // rule includes them.
    std::vector<int> request_human_index;
    std::vector<luce::common::PFlashTokenSpan> request_recent;
    std::vector<int> request_recent_index;
    // cand_v5 / stage 2: every normalised message (role, content span), and why the server could not map them.
    std::vector<luce::pflash::PFlashRequestMessage> request_messages_all;
    std::string request_messages_all_error;
    // Trace only: what the request probe did on this call (not owned; set
    // by the scorer for the selection it calls).
    const PFlashRequestProbeTrace * request_probe_trace = nullptr;
    // Query step 1 (pflash_query_candidates.h): PFLASH_QUERY_CANDIDATES / PFLASH_QUERY_PIN and their knobs;
    // all off by default. Needs PFLASH_SELECT_QUERY_PROBE=1.
    PFlashQueryConfig query;
    // PFLASH_EVAL_REF_IDENT_QUERY=heur (strict selection, head
    // scorer, not with the request probe): the scorer query is the
    // identifiers of the last turns (pflash_ident_query.h), scored and not
    // pinned. The tokens come per request (ident_error: why none).
    PFlashIdentQuery query_ident = PFlashIdentQuery::Off;
    std::vector<int> ident_tokens;
    int ident_turns = 0;
    std::string ident_error;
    // Trace only (not owned), as request_probe_trace.
    const PFlashIdentQueryTrace * ident_query_trace = nullptr;
    // PFLASH_FALLBACK_SPREAD=T (strict selection; opt-in, unset = off): a
    // request whose relevance spread (pflash_relevance_spread) is >= T is
    // served by full prefill of the raw prompt (failure policy D5 path,
    // reason "spread"). 0 = off.
    double fallback_spread = 0.0;
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
const char * pflash_select_strategy_name(PFlashSelectStrategy strategy) noexcept;

// PFLASH_SELECT_STRATEGY=fold, read once: the drafter loads its vocabulary
// at load time only when fold may be requested.
bool pflash_fold_requested() noexcept;

// PFLASH_SELECT_QUERY_PROBE=1, read once at load: the drafter loads the
// request probe from PFLASH_REQUEST_PROBE_GGUF (or refuses to load).
// PFLASH_SELECT_QUERY_PROBE_STUB=1 (testing): an all-zero request probe
// with tau = +inf replaces the GGUF, so every request falls back to
// today's windows through the probe path.
bool pflash_query_probe_requested() noexcept;
bool pflash_query_probe_stub_requested() noexcept;

// PFLASH_EVAL_REF_IDENT_QUERY, read per call: off (default, also when unset),
// or heur. Invalid values fail resolve_pflash_selection.
PFlashIdentQuery pflash_query_ident_requested() noexcept;

// Presence, rather than validity, gates cache and continuation policy so an
// empty or invalid experiment variable cannot silently fall back to legacy.
bool has_pflash_selection_environment() noexcept;

bool resolve_pflash_selection(
    int input_tokens,
    int legacy_chunk_size,
    PFlashSelectionConfig & out,
    std::string & error);

} // namespace luce::pflash
