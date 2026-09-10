#include "pflash_selection.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace dflash::qwen3 {

namespace {

constexpr const char * kModeEnv = "PFLASH_LONGATTNCOMP_MODE";
constexpr const char * kChunkEnv = "PFLASH_LONGATTNCOMP_CHUNK_SIZE";
constexpr const char * kQueryEnv = "PFLASH_LONGATTNCOMP_QUERY_TOKENS";
constexpr const char * kQueryParserEnv = "PFLASH_LONGATTNCOMP_QUERY_PARSER";
constexpr const char * kTopPEnv = "PFLASH_LONGATTNCOMP_TOP_P";

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

bool has_pflash_longattncomp_environment() noexcept {
    return std::getenv(kModeEnv) != nullptr ||
           std::getenv(kChunkEnv) != nullptr ||
           std::getenv(kQueryEnv) != nullptr ||
           std::getenv(kQueryParserEnv) != nullptr ||
           std::getenv(kTopPEnv) != nullptr;
}

bool pflash_chunk_is_structurally_required(
        int begin,
        int end,
        int query_begin,
        int query_end,
        int input_tokens,
        const std::vector<dflash::common::PFlashTokenSpan> &
            required_instruction_spans) noexcept {
    if (begin < 0 || end <= begin || query_begin < 0 ||
        query_end < query_begin || input_tokens < query_end ||
        end > input_tokens) {
        return false;
    }
    const bool query_chunk = begin < query_end && end > query_begin;
    const bool structural_suffix_chunk =
        begin < input_tokens && end > query_end;
    if (query_chunk || structural_suffix_chunk) return true;
    for (const auto & span : required_instruction_spans) {
        if (begin < span.end && end > span.begin) return true;
    }
    return false;
}

bool validate_pflash_instruction_spans(
        const std::vector<dflash::common::PFlashTokenSpan> & spans,
        int input_tokens,
        std::string & error) noexcept {
    error.clear();
    if (input_tokens < 0) {
        error = "PFlash input token count must not be negative";
        return false;
    }
    if (spans.size() > dflash::common::kPFlashMaxInstructionSpans) {
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

    for (const auto * candidate : optional) {
        if (mode == PFlashSelectionMode::CumulativeTopP &&
            result.retained_mass >= policy.top_p) {
            result.stop = PFlashSelectionStop::TopPReached;
            break;
        }

        const int length = candidate->end - candidate->begin;
        if (length > policy.token_budget - result.retained_tokens) {
            result.stop = PFlashSelectionStop::BudgetReached;
            break;
        }

        selected_candidates.push_back(candidate);
        result.retained_tokens += length;
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
    }
    return "unknown";
}

const char * pflash_selection_stop_name(PFlashSelectionStop stop) noexcept {
    switch (stop) {
        case PFlashSelectionStop::TopPReached: return "top_p_reached";
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

bool resolve_pflash_longattncomp(
        int input_tokens,
        int legacy_chunk_size,
        PFlashLongAttnCompConfig & out,
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

    PFlashLongAttnCompConfig config;
    config.configured = mode_raw || chunk_raw || query_raw ||
        query_parser_raw || top_p_raw;
    config.chunk_size = legacy_chunk_size;

    if (mode_raw) {
        if (std::strcmp(mode_raw, "budget_only") == 0) {
            config.mode = PFlashSelectionMode::BudgetOnly;
        } else if (std::strcmp(mode_raw, "top_p") == 0) {
            config.mode = PFlashSelectionMode::CumulativeTopP;
        } else {
            error = std::string(kModeEnv) + " must be budget_only or top_p";
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

    out = config;
    return true;
}

} // namespace dflash::qwen3
