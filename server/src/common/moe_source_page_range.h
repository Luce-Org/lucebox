#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace dflash::common {

struct MoeSourcePageRange {
    uintptr_t address = 0;
    size_t size = 0;
};

// Select only complete pages inside BOTH a tensor and its verified read-only
// file mapping. An empty interior is valid; invalid/overflowing spans fail.
inline bool moe_source_page_range(uintptr_t mapping, size_t mapping_size,
                                  uintptr_t tensor, size_t tensor_size,
                                  size_t page_size, MoeSourcePageRange & out) {
    out = {};
    const uintptr_t max = std::numeric_limits<uintptr_t>::max();
    if (!mapping || !page_size || (page_size & (page_size - 1)) != 0 ||
        mapping_size > max - mapping || tensor < mapping) return false;
    const uintptr_t mapping_end = mapping + mapping_size;
    if (tensor > mapping_end || tensor_size > mapping_end - tensor) return false;
    const size_t padding = (page_size - tensor % page_size) % page_size;
    if (padding > tensor_size) return true;
    const size_t length = ((tensor_size - padding) / page_size) * page_size;
    if (length) {
        out.address = tensor + padding;
        out.size = length;
    }
    return true;
}

inline bool moe_source_pageout_eligible(bool cold_gpu, bool hot_materialized,
                                       bool cold_materialized, bool cold_allocated) {
    return cold_gpu && hot_materialized && cold_materialized && cold_allocated;
}

}  // namespace dflash::common
