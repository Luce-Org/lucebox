// Unified cache-capacity helpers for server CLI/config.

#pragma once

#include <cstddef>
#include <string>

namespace dflash::common {

// The backend still allocates snapshots by small integer handles. User-facing
// config is byte-based; this estimate converts RAM budgets into an upper bound
// on simultaneously live handles. Actual RAM bytes are tracked after snapshot
// save and LRU-evicted against the configured budget.
inline constexpr size_t kCacheRamSlotEstimateBytes =
    (size_t)64 * 1024 * 1024;

// ModelBackend::kMaxSlots is 64 and the HTTP server reserves slot 63 as the
// disk-load staging slot, so cache pools may use at most 0..62.
inline constexpr int kCacheMaxRamSlots = 63;

inline constexpr size_t kDefaultCacheRamBytes =
    (size_t)1024 * 1024 * 1024;
inline constexpr size_t kDefaultCachePrefixRamBytes =
    (size_t)256 * 1024 * 1024;
inline constexpr size_t kDefaultCachePrefillRamBytes =
    kDefaultCacheRamBytes - kDefaultCachePrefixRamBytes;

inline constexpr size_t kDefaultCacheDiskBytes =
    (size_t)16 * 1024 * 1024 * 1024;
inline constexpr size_t kDefaultCachePrefixDiskBytes =
    (size_t)4 * 1024 * 1024 * 1024;
inline constexpr size_t kDefaultCachePrefillDiskBytes =
    kDefaultCacheDiskBytes - kDefaultCachePrefixDiskBytes;

struct CachePoolBudgets {
    size_t prefix_bytes = 0;
    size_t prefill_bytes = 0;
};

bool parse_cache_size_bytes(const std::string & text, size_t & out_bytes);
std::string format_cache_size(size_t bytes);
int cache_slots_for_ram_budget(size_t bytes, int max_slots = kCacheMaxRamSlots);
CachePoolBudgets split_cache_budget(size_t total_bytes,
                                    size_t default_prefix_bytes);
CachePoolBudgets resolve_cache_pool_budgets(size_t total_bytes,
                                            size_t default_prefix_bytes,
                                            bool total_seen,
                                            bool prefix_seen,
                                            size_t prefix_bytes,
                                            bool prefill_seen,
                                            size_t prefill_bytes);
std::string cache_subdir(const std::string & root, const char * name);

}  // namespace dflash::common
