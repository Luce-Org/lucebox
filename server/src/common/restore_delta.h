#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace dflash::common {

inline std::vector<int32_t> restore_prompt_delta(const std::vector<int32_t> & prompt,
                                                 int cached_prefix_len) {
    if (cached_prefix_len < 0) {
        throw std::invalid_argument("cached_prefix_len must be non-negative");
    }
    if (cached_prefix_len > (int)prompt.size()) {
        throw std::out_of_range("cached prefix is longer than prompt");
    }
    return std::vector<int32_t>(
        prompt.begin() + cached_prefix_len,
        prompt.end());
}

}  // namespace dflash::common
