#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace luce::common {

inline constexpr size_t kPFlashMaxInstructionSpans = 64;

// Half-open token range in the drafter-tokenized prompt.
struct PFlashTokenSpan {
    int begin = 0;
    int end = 0;
};

inline bool operator==(
        const PFlashTokenSpan & left,
        const PFlashTokenSpan & right) noexcept {
    return left.begin == right.begin && left.end == right.end;
}

inline bool operator!=(
        const PFlashTokenSpan & left,
        const PFlashTokenSpan & right) noexcept {
    return !(left == right);
}

// PFLASH_SELECT_QUERY_PROBE=1: the content spans of the human user
// messages (role "user" after normalisation; tool results are role "tool"),
// latest first, at most two ({0, 0} for one that renders empty).
// PFLASH_EVAL_REF_IDENT_QUERY=heur travels with them: the
// identifier tokens of the last turns (ascending positions), the turns read,
// and why the server found none it could map (``ident_error``).
// Candidate-rule inputs (read only by a GGUF with that candidate rule):
// ``human_index`` = the normalised message indices of ``human``; ``recent`` /
// ``recent_index`` = the content spans and indices of every non-system
// message, latest first.
struct PFlashRequestMessages {
    std::vector<PFlashTokenSpan> human;
    std::vector<int> human_index;
    std::vector<PFlashTokenSpan> recent;
    std::vector<int> recent_index;
    // cand_v4 / stage 2 (STAGE2-CONTRACT.md section 1): every normalised message in order, its role
    // ("system", "user", "assistant", "tool") and its content span (the tokens overlapping its exact content
    // characters; {0, 0} = no tokens); ``all_error`` = why the server could not map them.
    std::vector<std::string> all_roles;
    std::vector<PFlashTokenSpan> all_spans;
    std::string all_error;
    std::vector<int> ident_tokens;
    int ident_turns = 0;
    std::string ident_error;
};

} // namespace luce::common
