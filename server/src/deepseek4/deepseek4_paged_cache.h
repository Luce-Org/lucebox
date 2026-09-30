#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace luce::common {

// Per-layer cache geometry. Pure host data so the plan-only build can use it.
//   ratio 0            raw window only.
//   V4 ratio 4         CSA compressor: double-width state over the previous
//                      and current window (2*ratio rows), APE, plus its own
//                      indexer compressor feeding the index-key rows.
//   V4 ratio 128       HCA compressor: one window of `ratio` rows, single width.
//   V4.1 ratio 2       same pooling as HCA without APE; index keys derive from
//                      the latent, so index rows exist without indexer state.
//   V4.1 ratio 1       one latent per token: rows without any compressor state.
// Only kv source layers own compressed rows; V4.1 readers keep their ratio but
// allocate nothing beyond the raw ring.
struct DeepSeek4LayerGeometry {
    uint32_t ratio = 0;
    int64_t  head_dim = 0;             // raw / compressed row width (F16)
    int64_t  raw_rows = 0;             // n_swa
    bool     is_kv_source = false;     // runs the compressor; the only writer of comp_kv
    bool     is_index_source = false;  // scores index keys (owns a top-k)
    bool     has_comp = false;         // comp_kv rows (ratio > 0 at a kv source)
    int64_t  comp_width = 0;           // attn compressor state width (F32); 0 = stateless
    int64_t  comp_state_rows = 0;
    bool     has_index = false;        // index_comp_kv rows
    int64_t  index_dim = 0;            // index-key row width (F16)
    int64_t  index_state_width = 0;    // V4 indexer compressor state width (F32); 0 = none
    int64_t  index_state_rows = 0;
    bool has_comp_state() const { return has_comp && comp_state_rows > 0; }
    bool has_index_state() const { return has_index && index_state_rows > 0; }
    // Compressed-row capacity for a cache of `max_ctx` tokens (0 if !has_comp).
    int64_t comp_capacity(int max_ctx) const {
        return has_comp ? (int64_t) max_ctx / (int64_t) ratio + 16 : 0;
    }
};

DeepSeek4LayerGeometry deepseek4_layer_geometry_from_ratio(
    uint32_t ratio, int64_t head_dim, int64_t raw_rows, int64_t indexer_head_dim,
    bool is_kv_source, bool is_index_source);

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

// One geometry per layer; ratios must page (0, 1, 2, 4 or 128). A layer
// without compressed storage (ratio zero, or a V4.1 reader) has a raw ring
// but no compressed rows or compressor state.
bool plan_deepseek4_paged_cache(uint32_t head_dim,
                                uint32_t indexer_head_dim,
                                uint32_t slots,
                                uint32_t max_ctx,
                                uint32_t physical_blocks,
                                const std::vector<DeepSeek4LayerGeometry> & layers,
                                DeepSeek4PagedCachePlan & out);

// V4 form of the above: every compressing layer owns its rows and the ratio-4
// layers carry the indexer.
bool plan_deepseek4_paged_cache(uint32_t head_dim,
                                uint32_t indexer_head_dim,
                                uint32_t slots,
                                uint32_t max_ctx,
                                uint32_t physical_blocks,
                                const std::vector<uint32_t> & ratios,
                                DeepSeek4PagedCachePlan & out);

} // namespace luce::common
