// PFlash scoring pipeline glue: everything between per-token scores and the
// compressed id list.
//
//   - env_int / env_float      process knobs used across the drafter
//   - count_nonfinite_scores   score post-processing helper
//   - PFlashTraceFields / write_compression_trace
//                              JSONL compression trace (PFLASH_TRACE_PATH)
//   - select_pflash_chunks     per-token scores -> candidates -> strict
//                              selection -> merged output ids (+ trace)

#pragma once

#include "pflash_selection.h"
#include "pflash_seg_reranker.h"
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

inline size_t count_nonfinite_scores(const float * values, size_t count) {
    size_t nonfinite = 0;
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) ++nonfinite;
    }
    return nonfinite;
}

struct PFlashTraceFields {
    const std::vector<int32_t> * input_ids = nullptr;
    int query_begin = -1;
    int query_end = -1;
    luce::pflash::PFlashSelectionMode selector_mode =
        luce::pflash::PFlashSelectionMode::BudgetOnly;
    luce::pflash::PFlashQueryParser query_parser =
        luce::pflash::PFlashQueryParser::ArbitraryTail;
    int token_budget = 0;
    luce::pflash::PFlashSelectionStop stop =
        luce::pflash::PFlashSelectionStop::InvalidInput;
    int retained_tokens = 0;
    double retained_mass = 0.0;
    const std::vector<double> * exact_chunk_scores = nullptr;
    // Raw Stage-1 (pre-reranker) chunk scores when the segment reranker
    // overrode chunk_scores; absent otherwise.
    const std::vector<double> * stage1_chunk_scores = nullptr;
    const std::vector<PFlashTokenSpan> * required_instruction_spans = nullptr;
    // Variable-length candidates (segment probe): spans in candidate order.
    const std::vector<PFlashTokenSpan> * segments = nullptr;
    const char * segmentation = "probe";
    const char * candidate_score = "density";
    int top_k = 0;
    // Request probe + relative query candidates record for this call.
    const luce::pflash::PFlashRequestProbeTrace * request_probe = nullptr;
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
// strict selection.
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
// raw Stage-1 attention lift (mean per-token mass relative to uniform
// attention over the input; 1 = average, calibrated for recall gating) and
// its final ranking score (reranker-adjusted when a segment reranker is
// loaded; otherwise equal to lift). Cleared with the kept spans.
struct PFlashCandidateLift {
    PFlashTokenSpan span;
    double lift = 0.0;
    double rank_score = 0.0;
};
const std::vector<PFlashCandidateLift> & pflash_last_candidate_lifts();

// The pins the request probe added to the last strict selection on this
// thread: the query tokens it picked and its recency pin (not the caller's
// pins, nor the top-unit verbatim pin). A caller that selects by the
// candidate lifts itself keeps them. Cleared with the kept spans.
const std::vector<PFlashTokenSpan> & pflash_last_probe_pins();
void pflash_set_probe_pins(std::vector<PFlashTokenSpan> pins);

// Non-empty when a loaded segment reranker failed to execute during the last
// strict scoring call on this thread (distinct from the expected skip when
// fewer than two optional segments are eligible). Cleared with the kept spans.
const std::string & pflash_last_reranker_error();
void pflash_set_reranker_error(std::string error);

// Identity string of the drafter and PFlash asset files actually loaded for
// the last strict scoring call on this thread. Cleared with the kept spans.
const std::string & pflash_last_loaded_assets_identity();
void pflash_set_loaded_assets_identity(std::string identity);

// Sink-free attention concentration over optional (non-mandatory) candidates:
//   (selected_optional_mass / total_optional_mass) / (selected_optional_tokens / total_optional_tokens)
// excluding token 0 (attention sink). Returns 0.0 when degenerate.
double pflash_droppable_mass_lift(
        const std::vector<luce::pflash::PFlashSelectionCandidate> & candidates,
        const std::vector<float> & token_scores,
        const std::vector<uint8_t> & selected);

// The segment reranker's adaptive keep-ratio budget from the last strict selection on this
// thread; invalid when no reranker ran. Cleared with the kept spans.
const PFlashAdaptiveBudget & pflash_last_adaptive_budget();

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
        const std::vector<float> * rerank_scores = nullptr);

} // namespace luce::common
