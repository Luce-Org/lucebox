// The pager's configuration types, without the pager (no ggml): the pure slot planner builds on these.

#pragma once

#include <cstdint>

namespace luce::common {

constexpr int kKvFlashChunkTokens = 64;   // logical tokens per page, the default everywhere

struct KvFlashConfig {
    int chunk_tokens       = kKvFlashChunkTokens;  // logical tokens per page
    int pool_tokens        = 0;   // resident pool capacity (multiple of chunk_tokens)
    int sink_chunks        = 1;   // leading chunks never evicted (attention sinks)
    int tail_window_chunks = 4;   // trailing chunks never evicted (local window)
    int row_base           = 0;   // first row of this pool inside a shared KV allocation (slot slice)
    int victim_blocks      = 0;   // victim tier beside the pool (0 = off): paged-out chunks keep their bytes there, on the
                                  // device on GPU builds, so a recall of a chunk still in it is a D2D copy, not an H2D
};

// Minimum pool for a config: sinks + trailing window stay resident
// unconditionally, so at least 2 more chunks are required (1 evictable
// victim + the partially filled append head) or eviction deadlocks and
// slot_for() starts failing once the pool fills.
inline int kvflash_min_pool_tokens(const KvFlashConfig & cfg) {
    return (cfg.sink_chunks + cfg.tail_window_chunks + 2) * cfg.chunk_tokens;
}

// VRAM budget for "auto" pool sizing. Backends fill this AFTER the target
// weights are on the GPU and BEFORE the cache is allocated, so free_bytes
// reflects what the pool can actually use.
struct KvFlashAutoBudget {
    int64_t free_bytes      = 0;   // device free memory right now
    int64_t reserve_bytes   = 0;   // compute buffers + (if expected) drafter
    int64_t bytes_per_token = 0;   // pooled attention KV density for this model
    // Decode cost grows with the FA span (= the pool), so cap the auto pool
    // where speed stays near the small-pool point. Measured on the 27B/3090:
    // 1K pool 39.6 tok/s, 4K 38.7; 16K extrapolates to ~31-33, still 1.7-2.4x
    // the full cache at 128-256K. Override: LUCE_KVFLASH_MAX_POOL.
    int     speed_cap_tokens = 16384;
};

}  // namespace luce::common
