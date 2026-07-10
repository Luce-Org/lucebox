#pragma once

#include <cstddef>
#include <cstdint>

namespace dflash::common {

using Qwen35SplitPureChainHook = bool (*)(void * context);

// The only retained layer-split tree seam is the validated root-inclusive
// pure chain: slot 0 is the synthetic root and every later slot follows its
// immediate predecessor. The hook makes ordering observable in CPU-only tests.
inline bool qwen35_split_run_if_root_inclusive_pure_chain(
        const int32_t * parents,
        std::size_t parent_count,
        std::size_t n_actual,
        Qwen35SplitPureChainHook hook = nullptr,
        void * hook_context = nullptr) {
    if (parents == nullptr || n_actual == 0 || parent_count < n_actual) {
        return false;
    }
    if (parents[0] != -1) {
        return false;
    }
    for (std::size_t slot = 1; slot < n_actual; ++slot) {
        if (parents[slot] != static_cast<int32_t>(slot - 1)) {
            return false;
        }
    }
    return hook == nullptr || hook(hook_context);
}

}  // namespace dflash::common
