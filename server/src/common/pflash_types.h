#pragma once

#include <cstddef>

namespace dflash::common {

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

} // namespace dflash::common
