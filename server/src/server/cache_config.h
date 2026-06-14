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

inline constexpr size_t kDefaultPrefixCacheRamBytes =
    (size_t)32 * kCacheRamSlotEstimateBytes;

bool parse_cache_size_bytes(const std::string & text, size_t & out_bytes);
std::string format_cache_size(size_t bytes);
int cache_slots_for_ram_budget(size_t bytes, int max_slots = kCacheMaxRamSlots);
std::string cache_subdir(const std::string & root, const char * name);

}  // namespace dflash::common
