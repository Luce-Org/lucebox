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

struct PFlashLongAttnCompConfig {
    PFlashSelectionMode mode = PFlashSelectionMode::Legacy;
    PFlashQueryParser query_parser = PFlashQueryParser::SemanticUser;
    int chunk_size = 0;
    int query_tokens = 8;
    double top_p = 0.95;
    bool configured = false;
    bool selection_active = false;
};

// Presence, rather than validity, gates cache and continuation policy so an
// empty or invalid experiment variable cannot silently fall back to legacy.
bool has_pflash_longattncomp_environment() noexcept;

bool resolve_pflash_longattncomp(
    int input_tokens,
    int legacy_chunk_size,
    PFlashLongAttnCompConfig & out,
    std::string & error);

} // namespace dflash::qwen3
