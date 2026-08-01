// Profile-guided, model-neutral warm-cache planning for streamed MoE experts.

#pragma once

#include "moe_hybrid_routing_stats.h"
#include "moe_hybrid_stream.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace dflash::common {

enum class MoeStreamCacheOwner {
    All,
    Primary,
    Secondary,
};

struct MoeStreamCachePlanConfig {
    size_t max_entries = 0;
    // Zero means entry-count budgeting only. A nonzero budget makes the
    // planner rank by observed frequency per byte and skip entries that do not
    // fit the remaining capacity.
    uint64_t max_bytes = 0;
    MoeStreamCacheOwner owner = MoeStreamCacheOwner::All;
};

// Select a deterministic highest-value warm set. layer_expert_bytes contains
// one complete routed expert size per layer. Owner filtering reuses the exact
// runtime partition policy, so warming cannot duplicate an expert across the
// two Lucebox GPUs.
bool build_moe_stream_cache_plan(
    const MoeHybridRoutingStats & stats,
    const std::vector<uint64_t> & layer_expert_bytes,
    const MoeStreamCachePlanConfig & config,
    const MoeStreamDualOwnerPolicy * owner_policy,
    std::vector<MoeStreamCacheWarmEntry> & out,
    std::string * err = nullptr);

} // namespace dflash::common
