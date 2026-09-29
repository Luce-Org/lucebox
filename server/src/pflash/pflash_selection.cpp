#include "pflash_selection.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace luce::pflash {

namespace {

constexpr const char * kModeEnv = "PFLASH_SELECT_MODE";
constexpr const char * kChunkEnv = "PFLASH_SELECT_CHUNK_SIZE";
constexpr const char * kQueryEnv = "PFLASH_SELECT_QUERY_TOKENS";
constexpr const char * kQueryParserEnv = "PFLASH_SELECT_QUERY_PARSER";
constexpr const char * kTopPEnv = "PFLASH_SELECT_TOP_P";
constexpr const char * kTopKEnv = "PFLASH_SELECT_TOPK";
constexpr const char * kSegmentsEnv = "PFLASH_SELECT_SEGMENTS";
constexpr const char * kSelectEnv = "PFLASH_SELECT_SCORE";
constexpr const char * kScorerEnv = "PFLASH_SELECT_SCORER";
constexpr const char * kSplitEnv = "PFLASH_SELECT_SPLIT";

PFlashSelectionResult invalid_result(std::string error) {
    PFlashSelectionResult result;
    result.error = std::move(error);
    return result;
}

bool parse_int(const char * raw, int & out) {
    if (!raw || !*raw) return false;
    errno = 0;
    char * end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (errno == ERANGE || end == raw || *end != '\0' ||
        value < INT_MIN || value > INT_MAX) {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool parse_double(const char * raw, double & out) {
    if (!raw || !*raw) return false;
    errno = 0;
    char * end = nullptr;
    const double value = std::strtod(raw, &end);
    if (errno == ERANGE || end == raw || *end != '\0' ||
        !std::isfinite(value)) {
        return false;
    }
    out = value;
    return true;
}

int scheduled_chunk_size(int input_tokens) {
    if (input_tokens < 500) return 128;
    if (input_tokens < 3000) return 512;
    return 1024;
}

} // namespace

bool has_pflash_selection_environment() noexcept {
    return std::getenv(kModeEnv) != nullptr ||
           std::getenv(kChunkEnv) != nullptr ||
           std::getenv(kQueryEnv) != nullptr ||
           std::getenv(kQueryParserEnv) != nullptr ||
           std::getenv(kTopPEnv) != nullptr ||
           std::getenv(kTopKEnv) != nullptr ||
           std::getenv(kSegmentsEnv) != nullptr ||
           std::getenv(kSelectEnv) != nullptr ||
           std::getenv(kScorerEnv) != nullptr ||
           std::getenv(kSplitEnv) != nullptr;
}

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
    if (spans.size() > luce::common::kPFlashMaxInstructionSpans) {
        error = "PFlash has too many instruction spans";
        return false;
    }
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
        // Rank rule: K optional candidates in score order, the budget below
        // still a ceiling. K binding here means the budget never was.
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

    const char * mode_raw = std::getenv(kModeEnv);
    const char * chunk_raw = std::getenv(kChunkEnv);
    const char * query_raw = std::getenv(kQueryEnv);
    const char * query_parser_raw = std::getenv(kQueryParserEnv);
    const char * top_p_raw = std::getenv(kTopPEnv);
    const char * top_k_raw = std::getenv(kTopKEnv);
    const char * segments_raw = std::getenv(kSegmentsEnv);
    const char * select_raw = std::getenv(kSelectEnv);
    const char * scorer_raw = std::getenv(kScorerEnv);
    const char * split_raw = std::getenv(kSplitEnv);

    PFlashSelectionConfig config;
    config.configured = mode_raw || chunk_raw || query_raw ||
        query_parser_raw || top_p_raw || top_k_raw || segments_raw ||
        select_raw || scorer_raw || split_raw;
    if (scorer_raw) {
        if (std::strcmp(scorer_raw, "head") == 0) {
            config.scorer = PFlashScorer::Head;
        } else if (std::strcmp(scorer_raw, "legacy") == 0) {
            config.scorer = PFlashScorer::Legacy;
        } else if (std::strcmp(scorer_raw, "split") == 0) {
            config.scorer = PFlashScorer::Split;
        } else {
            error = std::string(kScorerEnv) + " must be head, legacy or split";
            return false;
        }
    }
    if (split_raw) {
        char * end = nullptr;
        errno = 0;
        const double value = std::strtod(split_raw, &end);
        if (errno != 0 || end == split_raw || *end != '\0' || !(value > 0.0 && value < 1.0)) {
            error = std::string(kSplitEnv) + " must be a fraction in (0, 1)";
            return false;
        }
        config.split_fraction = value;
    }
    if (segments_raw) {
        if (std::strcmp(segments_raw, "fixed") == 0) {
            config.segmentation = PFlashSegmentation::Fixed;
        } else if (std::strcmp(segments_raw, "probe") == 0) {
            config.segmentation = PFlashSegmentation::Probe;
        } else if (std::strcmp(segments_raw, "auto") != 0) {
            error = std::string(kSegmentsEnv) + " must be auto, fixed or probe";
            return false;
        }
    }
    if (select_raw) {
        if (std::strcmp(select_raw, "sum") == 0) {
            config.candidate_score = PFlashCandidateScore::Sum;
        } else if (std::strcmp(select_raw, "density") == 0) {
            config.candidate_score = PFlashCandidateScore::Density;
        } else if (std::strcmp(select_raw, "auto") != 0) {
            error = std::string(kSelectEnv) + " must be auto, sum or density";
            return false;
        }
    }
    config.chunk_size = legacy_chunk_size;

    if (mode_raw) {
        if (std::strcmp(mode_raw, "budget_only") == 0) {
            config.mode = PFlashSelectionMode::BudgetOnly;
        } else if (std::strcmp(mode_raw, "top_p") == 0) {
            config.mode = PFlashSelectionMode::CumulativeTopP;
        } else if (std::strcmp(mode_raw, "top_k") == 0) {
            config.mode = PFlashSelectionMode::TopK;
        } else {
            error = std::string(kModeEnv) +
                " must be budget_only, top_p or top_k";
            return false;
        }
    }
    config.selection_active = config.mode != PFlashSelectionMode::Legacy;

    if (chunk_raw) {
        if (!parse_int(chunk_raw, config.chunk_size) || config.chunk_size <= 0) {
            error = std::string(kChunkEnv) + " must be a positive integer";
            return false;
        }
    } else if (config.selection_active) {
        config.chunk_size = scheduled_chunk_size(input_tokens);
    }

    if (query_raw &&
        (!parse_int(query_raw, config.query_tokens) ||
         config.query_tokens < 1 || config.query_tokens > 512)) {
        error = std::string(kQueryEnv) + " must be an integer in [1, 512]";
        return false;
    }

    if (query_parser_raw) {
        if (std::strcmp(query_parser_raw, "latest_user") == 0) {
            config.query_parser = PFlashQueryParser::SemanticUser;
        } else if (std::strcmp(query_parser_raw, "arbitrary_tail") == 0) {
            config.query_parser = PFlashQueryParser::ArbitraryTail;
        } else {
            error = std::string(kQueryParserEnv) +
                " must be latest_user or arbitrary_tail";
            return false;
        }
    }

    if (top_p_raw &&
        (!parse_double(top_p_raw, config.top_p) ||
         config.top_p <= 0.0 || config.top_p > 1.0)) {
        error = std::string(kTopPEnv) + " must be finite and in (0, 1]";
        return false;
    }

    if (top_k_raw && (!parse_int(top_k_raw, config.top_k) || config.top_k <= 0)) {
        error = std::string(kTopKEnv) + " must be a positive integer";
        return false;
    }
    if (config.mode == PFlashSelectionMode::TopK && config.top_k <= 0) {
        error = std::string(kTopKEnv) + " is required when " +
            std::string(kModeEnv) + " is top_k";
        return false;
    }

    out = config;
    return true;
}

std::vector<luce::common::PFlashTokenSpan> pflash_probe_segments(
        const std::vector<float> & boundary_scores,
        int input_tokens,
        float threshold,
        int min_segment,
        int max_segment,
        const std::vector<int> & forced_cuts,
        const std::vector<float> & split_scores) {
    using luce::common::PFlashTokenSpan;
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
            spans.push_back({begin, best});
            begin = best;
        }
        spans.push_back({begin, end});
    }
    return spans;
}

PFlashSelectionResult select_pflash_split(
        const std::vector<PFlashSelectionCandidate> & head,
        const std::vector<PFlashSelectionCandidate> & other,
        const PFlashSelectionPolicy & policy,
        double head_fraction,
        PFlashSelectionMode mode) {
    PFlashSelectionResult result;
    if (mode == PFlashSelectionMode::TopK) {
        // A per-pass K would keep up to 2K segments, which is not the rule the
        // mode names; fail closed rather than quietly double it.
        result.stop = PFlashSelectionStop::InvalidInput;
        result.error = "split selection does not support top_k";
        return result;
    }
    if (head.size() != other.size() || !(head_fraction > 0.0 && head_fraction < 1.0)) {
        result.stop = PFlashSelectionStop::InvalidInput;
        result.error = "split selection needs matching candidate lists and a fraction in (0, 1)";
        return result;
    }
    for (size_t i = 0; i < head.size(); ++i) {
        if (head[i].ordinal != other[i].ordinal || head[i].begin != other[i].begin ||
            head[i].end != other[i].end || head[i].mandatory != other[i].mandatory) {
            result.stop = PFlashSelectionStop::InvalidInput;
            result.error = "split selection candidate lists describe different spans";
            return result;
        }
    }
    PFlashSelectionPolicy first = policy;
    first.token_budget = static_cast<int>(policy.token_budget * head_fraction);
    // Mandatory spans must fit even when the head's share is small.
    int mandatory = 0;
    for (const auto & c : head) if (c.mandatory) mandatory += c.end - c.begin;
    first.token_budget = (std::max)(first.token_budget, (std::min)(mandatory, policy.token_budget));
    const PFlashSelectionResult pass1 = select_pflash_candidates(head, first, mode);
    if (!pass1.ok) return pass1;
    std::vector<uint8_t> taken(head.size(), 0);
    for (size_t ordinal : pass1.ordinals) {
        for (size_t i = 0; i < head.size(); ++i) if (head[i].ordinal == ordinal) taken[i] = 1;
    }
    std::vector<PFlashSelectionCandidate> rest;
    for (size_t i = 0; i < other.size(); ++i) {
        if (taken[i]) continue;
        PFlashSelectionCandidate c = other[i];
        c.mandatory = false;  // mandatory spans were charged in pass 1
        rest.push_back(c);
    }
    PFlashSelectionPolicy second = policy;
    second.token_budget = policy.token_budget - pass1.retained_tokens;
    const PFlashSelectionResult pass2 = second.token_budget > 0
        ? select_pflash_candidates(rest, second, mode)
        : PFlashSelectionResult{};
    result.ok = true;
    result.ordinals = pass1.ordinals;
    result.ordinals.insert(result.ordinals.end(), pass2.ordinals.begin(), pass2.ordinals.end());
    std::sort(result.ordinals.begin(), result.ordinals.end());
    result.retained_tokens = pass1.retained_tokens + pass2.retained_tokens;
    result.retained_mass = pass1.retained_mass;  // the head's normalised mass share
    result.stop = second.token_budget > 0 ? pass2.stop : pass1.stop;
    return result;
}

const char * pflash_scorer_name(PFlashScorer scorer) noexcept {
    switch (scorer) {
        case PFlashScorer::Head: return "head";
        case PFlashScorer::Legacy: return "legacy";
        case PFlashScorer::Split: return "split";
    }
    return "unknown";
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
