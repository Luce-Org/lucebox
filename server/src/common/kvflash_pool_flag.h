// Whether the operator requested any KVFlash pool, without the pager's ggml dependency.
#pragma once

#include <cstdlib>
#include <cstring>

namespace luce::common {

// "auto" is a real request even though its final size is not known until the backend measures VRAM.
inline bool kvflash_pool_requested(const char * value) {
    return value != nullptr &&
           (std::strcmp(value, "auto") == 0 || std::atoi(value) > 0);
}

}  // namespace luce::common
