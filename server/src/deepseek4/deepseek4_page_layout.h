// Host-side address geometry for DeepSeek V4's paged raw and compressed KV.
#pragma once

#include <cstdint>
#include <limits>

namespace dflash::common {

inline constexpr uint32_t DS4_PAGE_TOKENS = 128;

// Raw KV remains slot-indexed: every sequence reuses this 128-row ring.
inline constexpr uint32_t ds4_raw_ring_row(uint64_t logical_token) {
    return static_cast<uint32_t>(logical_token % DS4_PAGE_TOKENS);
}

// Number of physically paged compressed rows. Rejects unsupported ratios and
// arithmetic that cannot be represented by the row-index type.
inline bool ds4_compressed_page_capacity(uint64_t physical_blocks,
                                         uint32_t ratio,
                                         uint64_t & rows) {
    if (ratio != 4 && ratio != 128) return false;
    const uint64_t rows_per_block = DS4_PAGE_TOKENS / ratio;
    if (physical_blocks >
        std::numeric_limits<uint64_t>::max() / rows_per_block) {
        return false;
    }
    rows = physical_blocks * rows_per_block;
    return true;
}

// Computes the destination for a completed compression group. `emitted` is
// false between group boundaries and `row` is left unchanged in that case.
// Physical block IDs need not be contiguous.
inline bool ds4_compressed_page_row(uint64_t logical_token,
                                    uint64_t physical_block,
                                    uint32_t ratio,
                                    uint64_t & row,
                                    bool & emitted) {
    if (ratio != 4 && ratio != 128) return false;
    emitted = logical_token % ratio == ratio - 1;
    if (!emitted) return true;

    const uint64_t rows_per_block = DS4_PAGE_TOKENS / ratio;
    if (physical_block >
        std::numeric_limits<uint64_t>::max() / rows_per_block) {
        return false;
    }
    const uint64_t base = physical_block * rows_per_block;
    const uint64_t offset = (logical_token % DS4_PAGE_TOKENS) / ratio;
    if (base > std::numeric_limits<uint64_t>::max() - offset) return false;
    row = base + offset;
    return true;
}

}  // namespace dflash::common
