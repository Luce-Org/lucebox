#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dflash::common {

// Pure host-side allocation plan. Byte counts describe tensor payloads (ggml
// alignment/padding is deliberately excluded).
struct DeepSeek4PagedCachePlan {
    uint32_t slots = 0;
    uint32_t max_ctx = 0;
    uint32_t physical_blocks = 0;
    uint32_t max_blocks_per_sequence = 0;
    uint64_t raw_bytes = 0;
    uint64_t compressed_bytes = 0;
    uint64_t state_bytes = 0;
    uint64_t total_persistent_bytes = 0;
    std::vector<uint32_t> ratios;
    std::vector<uint64_t> physical_rows;
};

// Host metadata for the gathered-reference decode graph.  Rows are expressed
// in the flattened persistent tensors: raw rows are [slot, ring-row], while
// compressed rows use the physical page geometry from deepseek4_page_layout.h.
// A negative slot denotes a padding lane and consequently has no scatter rows.
struct DeepSeek4GatheredLaneRows {
    int32_t slot = -1;
    int64_t position = 0;
    std::vector<int64_t> raw_history;
    std::vector<int64_t> compressed_history;
    uint32_t raw_history_valid = 0;
    uint32_t compressed_history_valid = 0;
    int64_t raw_scatter = -1;
    int64_t compressed_scatter = -1;
    bool compressed_emitted = false;
};

// Convert the configured pool capacity into 128-token physical pages. A zero
// request selects max_ctx * slots; an explicit request is honored even when it
// is smaller than max_ctx.
bool plan_deepseek4_paged_pool_blocks(uint32_t max_ctx,
                                      uint32_t slots,
                                      uint64_t requested_tokens,
                                      uint32_t & physical_blocks);

// block_tables is lane-major with block_table_stride entries per lane.
// Physical block IDs may be fragmented and are validated against
// physical_blocks.  History excludes the current token; compressed history is
// in chronological group order.  Returns false for malformed active lanes.
bool prepare_deepseek4_gathered_lane_rows(
    const int32_t * slots,
    const int64_t * positions,
    uint32_t lanes,
    const int32_t * block_tables,
    uint32_t block_table_stride,
    uint32_t physical_blocks,
    uint32_t ratio,
    std::vector<DeepSeek4GatheredLaneRows> & out);

// Ratios must contain only 0, 4, or 128. A ratio-zero layer has a raw ring
// but no compressed storage or compressor state.
bool plan_deepseek4_paged_cache(uint32_t head_dim,
                                uint32_t indexer_head_dim,
                                uint32_t slots,
                                uint32_t max_ctx,
                                uint32_t physical_blocks,
                                const std::vector<uint32_t> & ratios,
                                DeepSeek4PagedCachePlan & out);

} // namespace dflash::common
