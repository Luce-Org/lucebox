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

// Request-probe message map: the content spans of the human user messages
// (role "user" after normalisation; tool results are role "tool"), latest
// first, at most two ({0, 0} for one that renders empty), plus every
// normalised message in order (role + content span).
struct PFlashRequestMessages {
    std::vector<PFlashTokenSpan> human;
    std::vector<int> human_index;
    std::vector<PFlashTokenSpan> recent;
    std::vector<int> recent_index;
    std::vector<std::string> all_roles;
    std::vector<PFlashTokenSpan> all_spans;
    std::string all_error;
    // Standalone human-only request-probe caller (e.g. pflash_daemon `request`),
    // valid even when `human` is empty; does not require the full message map.
    bool human_only = false;
};

} // namespace luce::common
