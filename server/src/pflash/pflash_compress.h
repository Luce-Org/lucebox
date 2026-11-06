// PFlash scoring pipeline glue: everything between per-token scores and the
// compressed id list.
//
//   - env_int / env_float      process knobs used across the drafter
//   - count_nonfinite_scores / scoring_head_mean_token_mass
//                              score post-processing helpers
//   - PFlashTraceFields / write_compression_trace
//                              JSONL compression trace (PFLASH_TRACE_PATH)
//   - select_pflash_chunks     per-token scores -> candidates -> strict
//                              selection -> merged output ids (+ trace);
//                              with PFLASH_SELECT_STRATEGY=fold the fold
//                              rule (pflash_fold.h) over the drafter's
//                              decoded prompt (``fold_vocab``)

#pragma once

#include "pflash_selection.h"
#include "common/pflash_types.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace luce::common {

class Tokenizer;

int env_int(const char * name, int fallback);
float env_float(const char * name, float def);
void force_chunk_neighborhood(std::vector<uint8_t> & forced, int n_chunks,
                              int chunk, int radius);

struct QueryCaptureSlice {
    int chunk_offset = 0;
    int query_offset = 0;
    int tokens = 0;

    bool valid() const { return tokens > 0; }
};

inline QueryCaptureSlice query_capture_slice(
        int query_start,
        int query_end,
        int chunk_start,
        int chunk_tokens) {
    const int chunk_end = chunk_start + chunk_tokens;
    const int overlap_start = query_start > chunk_start ? query_start : chunk_start;
    const int overlap_end = query_end < chunk_end ? query_end : chunk_end;
    if (overlap_start >= overlap_end) return {};
    return {
        overlap_start - chunk_start,
        overlap_start - query_start,
        overlap_end - overlap_start,
    };
}

inline size_t count_nonfinite_scores(const float * values, size_t count) {
    size_t nonfinite = 0;
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) ++nonfinite;
    }
    return nonfinite;
}

// Scoring-head token mass: mean over heads and query tokens of softmax
// probabilities laid out as ggml [n_keys, n_queries, n_heads] (ne0 fastest).
inline void scoring_head_mean_token_mass(
        const float * probs,
        int n_keys,
        int n_queries,
        int n_heads,
        std::vector<float> & out) {
    out.assign((size_t) n_keys, 0.0f);
    if (n_keys <= 0 || n_queries <= 0 || n_heads <= 0) return;
    std::vector<double> sum((size_t) n_keys, 0.0);
    for (int h = 0; h < n_heads; ++h) {
        for (int t = 0; t < n_queries; ++t) {
            const float * row = probs + ((size_t) h * n_queries + t) * n_keys;
            for (int j = 0; j < n_keys; ++j) sum[(size_t) j] += row[j];
        }
    }
    const double denominator = (double) n_heads * (double) n_queries;
    for (int j = 0; j < n_keys; ++j) out[(size_t) j] = (float) (sum[(size_t) j] / denominator);
}

struct PFlashTraceFields {
    const std::vector<int32_t> * input_ids = nullptr;
    int query_begin = -1;
    int query_end = -1;
    luce::pflash::PFlashSelectionMode selector_mode =
        luce::pflash::PFlashSelectionMode::Legacy;
    luce::pflash::PFlashQueryParser query_parser =
        luce::pflash::PFlashQueryParser::SemanticUser;
    int token_budget = 0;
    luce::pflash::PFlashSelectionStop stop =
        luce::pflash::PFlashSelectionStop::InvalidInput;
    int retained_tokens = 0;
    double retained_mass = 0.0;
    const std::vector<double> * exact_chunk_scores = nullptr;
    const std::vector<PFlashTokenSpan> * required_instruction_spans = nullptr;
    // Variable-length candidates (segment probe): spans in candidate order.
    const std::vector<PFlashTokenSpan> * segments = nullptr;
    const char * segmentation = "fixed";
    const char * candidate_score = "sum";
    // Two-scorer selection: the other scorer's candidate scores, same order.
    const char * scorer = "head";
    double split_fraction = 0.0;
    const std::vector<double> * other_chunk_scores = nullptr;
    // Rank-mode ceiling: the K that applied, 0 outside top_k mode.
    int top_k = 0;
    // Fold strategy: the kept spans (arbitrary token ranges), the route and
    // the structure it found; ``selected`` then marks the anchors taken.
    const std::vector<PFlashTokenSpan> * kept_spans = nullptr;
    const std::vector<PFlashTokenSpan> * fold_blocks = nullptr;
    bool fold_code = false;
    int fold_code_lines = 0;
    int fold_nonblank_lines = 0;
    int fold_code_blocks = 0;
    const char * fold_text_source = "none";
    int fold_skipped = 0;
    int fold_capped = 0;
    int fold_contained = 0;
    // Query head: the extra query windows scored next to the query.
    const std::vector<PFlashTokenSpan> * extra_queries = nullptr;
    // PFLASH_SELECT_QUERY_PROBE=1: the request probe's record of this call.
    const luce::pflash::PFlashRequestProbeTrace * request_probe = nullptr;
    // ``"ident_query":{...}`` (PFLASH_EVAL_REF_IDENT_QUERY only; absent otherwise).
    const luce::pflash::PFlashIdentQueryTrace * ident_query = nullptr;
    // PFLASH_FALLBACK_SPREAD on (threshold > 0): ``"spread":x,
    // "spread_threshold":T,"spread_first_token_score":s,"spread_per_token":b``
    // close the record (the inputs pflash_selection_spread adds to the
    // record's chunks); absent when off.
    double spread = 0.0;
    double spread_threshold = 0.0;
    double spread_first_token_score = 0.0;
    bool spread_per_token = false;
};

void write_compression_trace(
        int input_tokens,
        float keep_ratio,
        int chunk_size,
        int n_lookahead,
        int pool_kernel,
        int n_keep,
        const std::vector<std::pair<float, int>> & chunk_means,
        const std::vector<uint8_t> & selected,
        const std::vector<uint8_t> & forced,
        const std::vector<int32_t> & compressed_ids,
        const PFlashTraceFields * trace_fields = nullptr);

// The spans the last strict selection on this thread kept, in input
// coordinates, ascending and merged. Cleared at the start of every
// drafter_score_and_compress call; empty when the call did not reach a
// strict selection (legacy selection, errors).
const std::vector<PFlashTokenSpan> & pflash_last_kept_spans();
void pflash_clear_kept_spans();

// What the last strict scoring on this thread reused: the token its drafter
// session resumed from, how many tokens it ran, how many query windows it
// scored. Cleared with the kept spans.
struct PFlashScoringStats {
    int resume = -1;
    int new_tokens = -1;
    int query_windows = 0;
    double forward_s = 0.0;
};
const PFlashScoringStats & pflash_last_scoring_stats();
void pflash_set_scoring_stats(const PFlashScoringStats & stats);

// Stage 2's budget decision on the last strict scoring on this thread (STAGE2-CONTRACT.md section 5 step 5):
// ``widened`` when l_vague >= tau_fb (B' = min(n, B + B/2) drafter tokens), ``raw_fits`` when B' >= n (the
// caller serves the raw prompt). Cleared with the kept spans.
struct PFlashStage2Outcome {
    bool widened = false;
    bool raw_fits = false;
    int budget = -1;            // B
    int budget_widened = -1;    // B'
};
const PFlashStage2Outcome & pflash_last_stage2_outcome();
void pflash_set_stage2_outcome(const PFlashStage2Outcome & outcome);

// Every candidate of the last strict selection on this thread with its
// attention lift: mean per-token mass relative to uniform attention over
// the input (1 = average, 20 = twenty times average). Cleared with the
// kept spans.
struct PFlashCandidateLift {
    PFlashTokenSpan span;
    double lift = 0.0;
};
const std::vector<PFlashCandidateLift> & pflash_last_candidate_lifts();

// The pins the request probe added to the last strict selection on this
// thread: the query tokens it picked and its recency pin (not the caller's
// pins, nor PFLASH_QUERY_PIN's top units). A caller that selects by the
// candidate lifts itself keeps them. Cleared with the kept spans.
const std::vector<PFlashTokenSpan> & pflash_last_probe_pins();
void pflash_set_probe_pins(std::vector<PFlashTokenSpan> pins);

// Relevance spread of one selection (PFLASH_FALLBACK_SPREAD): with
// p_i = s_i / sum(s) over every candidate's score (the trace's
// ``chunk_scores``), H = -sum_{p_i > 0} p_i ln p_i and k = the selected
// candidates that are not forced (``selected_chunks`` minus
// ``forced_chunks``), spread = exp(H) / max(k, 1). A sum <= 0 gives H = 0
// (spread 1 / max(k, 1)). ``valid`` is false on mismatched sizes or a
// non-finite score; then no decision may be taken from it.
struct PFlashRelevanceSpread {
    bool valid = false;
    double spread = 0.0;
    double entropy = 0.0;
    int k = 0;              // selected, not forced
    int candidates = 0;
};
PFlashRelevanceSpread pflash_relevance_spread(
        const std::vector<double> & scores,
        const std::vector<uint8_t> & selected,
        const std::vector<uint8_t> & forced);
// The spread of a selection over its chunks, with the prompt's first token
// scored as a unit of its own. That token is the attention sink and holds most
// of the score; averaged into the chunk around it, the spread would depend on
// how long that chunk is, and so on the system prompt's length once the system
// turn is pinned whole. ``first_chunk_tokens`` is chunk 0's length,
// ``first_token_score`` the token's own score and ``per_token`` whether chunk
// scores are per-token means. The sink unit counts as forced, so k is the
// selection's own. Equals pflash_relevance_spread when chunk 0 is one token.
PFlashRelevanceSpread pflash_selection_spread(
        const std::vector<double> & chunk_scores,
        const std::vector<uint8_t> & selected,
        const std::vector<uint8_t> & forced,
        int first_chunk_tokens,
        double first_token_score,
        bool per_token);
// The spread of the last traced (served) strict selection on this thread;
// invalid when none ran. Cleared with the kept spans.
const PFlashRelevanceSpread & pflash_last_relevance_spread();

std::vector<int32_t> select_pflash_chunks(
        const std::vector<int32_t> & ids,
        const std::vector<float> & token_scores,
        float keep_ratio,
        int n_lookahead,
        int score_query_end,
        int pool_kernel,
        const luce::pflash::PFlashSelectionConfig & config,
        const std::vector<PFlashTokenSpan> & required_instruction_spans,
        bool direct_mass,
        bool write_trace,
        const std::vector<PFlashTokenSpan> * segments = nullptr,
        bool density = false,
        const std::vector<float> * other_token_scores = nullptr,
        double split_fraction = 0.0,
        const Tokenizer * fold_vocab = nullptr);

} // namespace luce::common
