// PFlash scoring pipeline glue: everything between per-token scores and the
// compressed id list.
//
//   - env_int / env_float      process knobs used across the drafter
//   - count_nonfinite_scores / scoring_head_mean_token_mass
//                              score post-processing helpers
//   - PFlashTraceFields / write_compression_trace
//                              JSONL compression trace (PFLASH_TRACE_PATH)
//   - select_pflash_chunks     per-token scores -> candidates -> strict
//                              selection -> merged output ids (+ trace)

#pragma once

#include "pflash_selection.h"
#include "common/pflash_types.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace luce::common {

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

// Every candidate of the last strict selection on this thread with its
// attention lift: mean per-token mass relative to uniform attention over
// the input (1 = average, 20 = twenty times average). Cleared with the
// kept spans.
struct PFlashCandidateLift {
    PFlashTokenSpan span;
    double lift = 0.0;
};
const std::vector<PFlashCandidateLift> & pflash_last_candidate_lifts();

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
        double split_fraction = 0.0);

} // namespace luce::common
