#pragma once
#include <algorithm>
#include <stdexcept>
namespace luce::common {
// Match the reference recurrent checkpoints: final chunk, then final four.
// The partition affects Vulkan fusion/reduction order; it is not just batching.
inline int lfm_prefill_partition(int remaining, int chunk) {
    if (remaining <= 0 || chunk <= 0) throw std::invalid_argument("invalid prefill size");
    return std::min(chunk, remaining > chunk ? remaining - chunk
                         : remaining > 4 ? remaining - 4 : remaining);
}
}
