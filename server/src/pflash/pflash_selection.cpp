#include "pflash_selection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace luce::pflash {

namespace {

PFlashSelectionResult invalid_result(std::string error) {
    PFlashSelectionResult result;
    result.error = std::move(error);
    return result;
}

} // namespace

bool pflash_chunk_is_structurally_required(
        int begin,
        int end,
        int query_begin,
        int query_end,
        int input_tokens,
        const std::vector<luce::common::PFlashTokenSpan> &
            required_instruction_spans,
        bool query_suffix_structural) noexcept {
    if (begin < 0 || end <= begin || query_begin < 0 ||
        query_end < query_begin || input_tokens < query_end ||
        end > input_tokens) {
        return false;
    }
    const bool query_chunk = begin < query_end && end > query_begin;
    const bool structural_suffix_chunk = query_suffix_structural &&
        begin < input_tokens && end > query_end;
    if (query_chunk || structural_suffix_chunk) return true;
    for (const auto & span : required_instruction_spans) {
        if (begin < span.end && end > span.begin) return true;
    }
    return false;
}

bool validate_pflash_instruction_spans(
        const std::vector<luce::common::PFlashTokenSpan> & spans,
        int input_tokens,
        std::string & error) noexcept {
    error.clear();
    if (input_tokens < 0) {
        error = "PFlash input token count must not be negative";
        return false;
    }
    // No count limit in process (an agent session pins one role header per
    // turn: hundreds of spans). kPFlashMaxInstructionSpans bounds the IPC
    // message only (pflash_drafter_ipc.cpp checks it there).
    int previous_end = 0;
    for (const auto & span : spans) {
        if (span.begin < 0 || span.end <= span.begin ||
            span.end > input_tokens) {
            error = "PFlash instruction span is outside the input";
            return false;
        }
        if (span.begin < previous_end) {
            error = "PFlash instruction spans must be ordered and non-overlapping";
            return false;
        }
        previous_end = span.end;
    }
    return true;
}

PFlashSelectionResult select_pflash_candidates(
        const std::vector<PFlashSelectionCandidate> & candidates,
        const PFlashSelectionPolicy & policy,
        PFlashSelectionMode mode) {
    if (mode == PFlashSelectionMode::Legacy) {
        return invalid_result("legacy mode does not use strict PFlash selection");
    }
    if (policy.token_budget <= 0) {
        return invalid_result("PFlash token budget must be positive");
    }
    if (!std::isfinite(policy.top_p) || policy.top_p <= 0.0 || policy.top_p > 1.0) {
        return invalid_result("PFlash top_p must be finite and in (0, 1]");
    }
    if (mode == PFlashSelectionMode::TopK && policy.top_k <= 0) {
        return invalid_result("PFlash top_k must be positive");
    }

    std::vector<const PFlashSelectionCandidate *> source_ranges;
    source_ranges.reserve(candidates.size());
    std::vector<size_t> ordinals;
    ordinals.reserve(candidates.size());
    for (const auto & candidate : candidates) {
        if (candidate.begin < 0 || candidate.end <= candidate.begin) {
            return invalid_result("PFlash candidate range is invalid");
        }
        if (!std::isfinite(candidate.score)) {
            return invalid_result("PFlash candidate score must be finite");
        }
        source_ranges.push_back(&candidate);
        ordinals.push_back(candidate.ordinal);
    }

    std::sort(ordinals.begin(), ordinals.end());
    if (std::adjacent_find(ordinals.begin(), ordinals.end()) != ordinals.end()) {
        return invalid_result("PFlash candidate ordinals must be unique");
    }
    std::sort(source_ranges.begin(), source_ranges.end(),
        [](const auto * left, const auto * right) {
            if (left->begin != right->begin) return left->begin < right->begin;
            return left->end < right->end;
        });
    for (size_t index = 1; index < source_ranges.size(); ++index) {
        if (source_ranges[index - 1]->end > source_ranges[index]->begin) {
            return invalid_result("PFlash candidate ranges must not overlap");
        }
    }

    PFlashSelectionResult result;
    result.ok = true;
    result.stop = PFlashSelectionStop::CandidatesExhausted;
    std::vector<const PFlashSelectionCandidate *> selected_candidates;
    selected_candidates.reserve(candidates.size());

    std::vector<const PFlashSelectionCandidate *> optional;
    optional.reserve(candidates.size());
    for (const auto & candidate : candidates) {
        if (candidate.mandatory) {
            const int length = candidate.end - candidate.begin;
            if (length > policy.token_budget - result.retained_tokens) {
                result = {};
                result.stop = PFlashSelectionStop::MandatoryQueryExceedsBudget;
                result.error = "mandatory PFlash retention tokens exceed the token budget";
                return result;
            }
            selected_candidates.push_back(&candidate);
            result.retained_tokens += length;
        } else {
            optional.push_back(&candidate);
        }
    }

    std::sort(optional.begin(), optional.end(),
        [](const auto * left, const auto * right) {
            const double left_score = std::max(0.0, left->score);
            const double right_score = std::max(0.0, right->score);
            if (left_score != right_score) return left_score > right_score;
            return left->ordinal < right->ordinal;
        });

    double max_score = 0.0;
    for (const auto * candidate : optional) {
        max_score = std::max(max_score, std::max(0.0, candidate->score));
    }
    double scaled_total = 0.0;
    if (max_score > 0.0) {
        for (const auto * candidate : optional) {
            scaled_total += std::max(0.0, candidate->score) / max_score;
        }
    }

    int kept_optional = 0;
    for (const auto * candidate : optional) {
        if (mode == PFlashSelectionMode::CumulativeTopP &&
            result.retained_mass >= policy.top_p) {
            result.stop = PFlashSelectionStop::TopPReached;
            break;
        }
        if (mode == PFlashSelectionMode::TopK && kept_optional >= policy.top_k) {
            result.stop = PFlashSelectionStop::TopKReached;
            break;
        }

        const int length = candidate->end - candidate->begin;
        if (length > policy.token_budget - result.retained_tokens) {
            result.stop = PFlashSelectionStop::BudgetReached;
            if (policy.skip_oversized) continue;
            break;
        }

        selected_candidates.push_back(candidate);
        result.retained_tokens += length;
        ++kept_optional;
        if (!optional.empty()) {
            result.retained_mass += max_score > 0.0
                ? (std::max(0.0, candidate->score) / max_score) / scaled_total
                : 1.0 / static_cast<double>(optional.size());
        }
    }

    std::sort(selected_candidates.begin(), selected_candidates.end(),
        [](const auto * left, const auto * right) {
            if (left->begin != right->begin) return left->begin < right->begin;
            return left->end < right->end;
        });
    result.ordinals.reserve(selected_candidates.size());
    for (const auto * candidate : selected_candidates) {
        result.ordinals.push_back(candidate->ordinal);
    }
    return result;
}

const char * pflash_selection_mode_name(PFlashSelectionMode mode) noexcept {
    switch (mode) {
        case PFlashSelectionMode::Legacy: return "legacy";
        case PFlashSelectionMode::BudgetOnly: return "budget_only";
        case PFlashSelectionMode::CumulativeTopP: return "top_p";
        case PFlashSelectionMode::TopK: return "top_k";
    }
    return "unknown";
}

const char * pflash_selection_stop_name(PFlashSelectionStop stop) noexcept {
    switch (stop) {
        case PFlashSelectionStop::TopPReached: return "top_p_reached";
        case PFlashSelectionStop::TopKReached: return "top_k_reached";
        case PFlashSelectionStop::BudgetReached: return "budget_reached";
        case PFlashSelectionStop::CandidatesExhausted: return "candidates_exhausted";
        case PFlashSelectionStop::InvalidInput: return "invalid_input";
        case PFlashSelectionStop::MandatoryQueryExceedsBudget:
            return "mandatory_query_exceeds_budget";
    }
    return "unknown";
}

const char * pflash_query_parser_name(PFlashQueryParser parser) noexcept {
    switch (parser) {
        case PFlashQueryParser::SemanticUser: return "latest_user";
        case PFlashQueryParser::ArbitraryTail: return "arbitrary_tail";
    }
    return "unknown";
}

bool resolve_pflash_selection(
        int input_tokens,
        int legacy_chunk_size,
        PFlashSelectionConfig & out,
        std::string & error) {
    error.clear();
    if (input_tokens < 0) {
        error = "PFlash input token count must not be negative";
        return false;
    }
    if (legacy_chunk_size <= 0) {
        error = "PFlash legacy chunk size must be positive";
        return false;
    }
    out = PFlashSelectionConfig{};
    return true;
}

bool resolve_pflash_drafter_selection(
        int input_tokens,
        int caller_chunk_size,
        const luce::common::PFlashRequestMessages * request_messages,
        PFlashSelectionConfig & out,
        std::string & error) {
    if (!resolve_pflash_selection(input_tokens, caller_chunk_size, out, error)) {
        return false;
    }
    if (!request_messages) {
        out.chunk_size = caller_chunk_size;
        out.segmentation = PFlashSegmentation::Auto;
        out.candidate_score = PFlashCandidateScore::Auto;
        out.query_probe = false;
        out.query.candidates = 0;
        out.query.pin_tokens = 0;
    } else if (request_messages->human_only) {
        out.chunk_size = caller_chunk_size;
        out.query.candidates = 0;
        out.query.pin_tokens = 0;
    }
    return true;
}

PFlashLineFlags pflash_line_flags(const std::vector<std::string> & token_texts) {
    PFlashLineFlags flags;
    const size_t n = token_texts.size();
    flags.line_start.assign(n, 0);
    flags.after_blank.assign(n, 0);
    // The text of the current line seen so far: is it only spaces / tabs / CR?
    bool line_blank = true;
    for (size_t t = 0; t < n; ++t) {
        const std::string & text = token_texts[t];
        const size_t last_nl = text.rfind('\n');
        if (last_nl != std::string::npos) {
            // The line ended by text's last '\n': the pending line content plus
            // this token's text after its previous '\n' (if any) up to last_nl.
            const size_t prev_nl = last_nl == 0 ? std::string::npos : text.rfind('\n', last_nl - 1);
            bool blank = prev_nl == std::string::npos ? line_blank : true;
            const size_t from = prev_nl == std::string::npos ? 0 : prev_nl + 1;
            for (size_t i = from; i < last_nl && blank; ++i) {
                const char c = text[i];
                if (c != ' ' && c != '\t' && c != '\r') blank = false;
            }
            if (t + 1 < n) {
                flags.line_start[t + 1] = 1;
                flags.after_blank[t + 1] = blank ? 1 : 0;
            }
            line_blank = true;
            for (size_t i = last_nl + 1; i < text.size(); ++i) {
                const char c = text[i];
                if (c != ' ' && c != '\t' && c != '\r') { line_blank = false; break; }
            }
        } else {
            for (const char c : text) {
                if (c != ' ' && c != '\t' && c != '\r') { line_blank = false; break; }
            }
        }
    }
    return flags;
}

std::vector<luce::common::PFlashTokenSpan> pflash_probe_segments(
        const std::vector<float> & boundary_scores,
        int input_tokens,
        float threshold,
        int min_segment,
        int max_segment,
        const std::vector<int> & forced_cuts,
        const std::vector<float> & split_scores) {
    return pflash_probe_segments(boundary_scores, input_tokens, threshold, min_segment,
                                 max_segment, forced_cuts, split_scores, nullptr);
}

std::vector<luce::common::PFlashTokenSpan> pflash_probe_segments(
        const std::vector<float> & boundary_scores,
        int input_tokens,
        float threshold,
        int min_segment,
        int max_segment,
        const std::vector<int> & forced_cuts,
        const std::vector<float> & split_scores,
        const PFlashLineFlags * lines) {
    using luce::common::PFlashTokenSpan;
    if (lines && ((int) lines->line_start.size() < input_tokens ||
                  (int) lines->after_blank.size() < input_tokens)) {
        return {};
    }
    std::vector<PFlashTokenSpan> spans;
    if (input_tokens <= 0 || (int) boundary_scores.size() < input_tokens ||
        min_segment < 1 || max_segment < min_segment) {
        return spans;
    }
    // Sub-unit scores feed only the oversize interior argmax;
    // the boundary threshold and merge floor always read the unit scores.
    const std::vector<float> & interior =
        (int) split_scores.size() >= input_tokens ? split_scores : boundary_scores;
    std::vector<uint8_t> forced((size_t) input_tokens + 1, 0);
    for (int cut : forced_cuts) {
        if (cut > 0 && cut < input_tokens) forced[(size_t) cut] = 1;
    }
    std::vector<int> cuts;
    cuts.push_back(0);
    for (int token = 1; token < input_tokens; ++token) {
        const bool wanted = forced[(size_t) token] ||
            (std::isfinite(boundary_scores[(size_t) token]) &&
             boundary_scores[(size_t) token] > threshold);
        if (!wanted) continue;
        if (!forced[(size_t) token] && token - cuts.back() < min_segment) continue;
        cuts.push_back(token);
    }
    cuts.push_back(input_tokens);
    for (size_t index = 1; index < cuts.size(); ++index) {
        int begin = cuts[index - 1];
        const int end = cuts[index];
        while (end - begin > max_segment) {
            // Split at the best-scoring interior token in the second half of
            // the next max_segment piece (the distance guard keeps the split
            // off the near edge), honoring the min_segment margins on both
            // sides; else on a fixed grid.
            const int lo = std::max(begin + min_segment, begin + max_segment / 2);
            const int hi = std::min(end - min_segment, begin + max_segment);
            int best = -1;
            float best_score = 0.0f;
            for (int token = lo; token <= hi; ++token) {
                const float score = interior[(size_t) token];
                if (std::isfinite(score) && score > best_score) {
                    best_score = score;
                    best = token;
                }
            }
            if (best < 0) best = begin + max_segment;
            if (lines) {
                // Line-aligned: blank-line starts first, then any line start;
                // the highest split score, earliest on ties.
                for (int pass = 0; pass < 2; ++pass) {
                    int aligned = -1;
                    for (int token = lo; token <= hi; ++token) {
                        if (!lines->line_start[(size_t) token]) continue;
                        if (pass == 0 && !lines->after_blank[(size_t) token]) continue;
                        if (aligned < 0 || interior[(size_t) token] > interior[(size_t) aligned]) {
                            aligned = token;
                        }
                    }
                    if (aligned >= 0) {
                        best = aligned;
                        break;
                    }
                }
            }
            spans.push_back({begin, best});
            begin = best;
        }
        spans.push_back({begin, end});
    }
    return spans;
}

const char * pflash_segmentation_name(PFlashSegmentation segmentation) noexcept {
    switch (segmentation) {
        case PFlashSegmentation::Auto: return "auto";
        case PFlashSegmentation::Fixed: return "fixed";
        case PFlashSegmentation::Probe: return "probe";
    }
    return "unknown";
}

const char * pflash_candidate_score_name(PFlashCandidateScore score) noexcept {
    switch (score) {
        case PFlashCandidateScore::Auto: return "auto";
        case PFlashCandidateScore::Sum: return "sum";
        case PFlashCandidateScore::Density: return "density";
    }
    return "unknown";
}

} // namespace luce::pflash
