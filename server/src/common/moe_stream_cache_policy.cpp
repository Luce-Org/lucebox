#include "moe_stream_cache_policy.h"

#include <algorithm>
#include <limits>
#include <new>

namespace dflash::common {

bool build_moe_stream_cache_plan(
        const MoeHybridRoutingStats & stats,
        const std::vector<uint64_t> & layer_expert_bytes,
        const MoeStreamCachePlanConfig & config,
        const MoeStreamDualOwnerPolicy * owner_policy,
        std::vector<MoeStreamCacheWarmEntry> & out,
        std::string * err) {
    out.clear();
    if (stats.empty() || stats.n_layer <= 0 || stats.n_expert <= 0 ||
        layer_expert_bytes.size() != (size_t) stats.n_layer) {
        if (err) *err = "invalid routing profile or per-layer expert sizes";
        return false;
    }
    if (config.max_entries == 0) return true;
    if (config.owner != MoeStreamCacheOwner::All) {
        if (!owner_policy || owner_policy->primary_share_per_mille < 0 ||
            owner_policy->primary_share_per_mille > 1000 ||
            (owner_policy->primary_placement &&
             (owner_policy->primary_placement->n_layer != stats.n_layer ||
              owner_policy->primary_placement->n_expert != stats.n_expert))) {
            if (err) *err = "cache owner policy does not match routing profile";
            return false;
        }
    }

    std::vector<MoeStreamCacheWarmEntry> candidates;
    try {
        candidates.reserve((size_t) stats.n_layer * (size_t) stats.n_expert);
    } catch (const std::bad_alloc &) {
        if (err) *err = "failed to allocate streamed cache candidates";
        return false;
    }
    for (int layer = 0; layer < stats.n_layer; ++layer) {
        const uint64_t bytes = layer_expert_bytes[(size_t) layer];
        if (bytes == 0) {
            if (err) *err = "streamed cache plan has a zero-sized expert";
            return false;
        }
        for (int expert = 0; expert < stats.n_expert; ++expert) {
            const uint64_t frequency = stats.count(layer, expert);
            if (frequency == 0) continue;
            if (config.owner != MoeStreamCacheOwner::All) {
                const bool primary = moe_stream_primary_owns_expert(
                    *owner_policy, layer, expert);
                if ((config.owner == MoeStreamCacheOwner::Primary) != primary) {
                    continue;
                }
            }
            candidates.push_back({
                (int32_t) layer, (int32_t) expert, frequency, bytes});
        }
    }

    std::stable_sort(candidates.begin(), candidates.end(),
        [](const MoeStreamCacheWarmEntry & a,
           const MoeStreamCacheWarmEntry & b) {
            const long double av = (long double) a.frequency /
                                   (long double) a.bytes;
            const long double bv = (long double) b.frequency /
                                   (long double) b.bytes;
            if (av != bv) return av > bv;
            if (a.frequency != b.frequency) return a.frequency > b.frequency;
            if (a.layer != b.layer) return a.layer < b.layer;
            return a.expert < b.expert;
        });

    uint64_t used_bytes = 0;
    out.reserve(std::min(config.max_entries, candidates.size()));
    for (const MoeStreamCacheWarmEntry & candidate : candidates) {
        if (out.size() >= config.max_entries) break;
        if (config.max_bytes != 0) {
            if (candidate.bytes > config.max_bytes -
                    std::min(config.max_bytes, used_bytes)) {
                continue;
            }
            used_bytes += candidate.bytes;
        }
        out.push_back(candidate);
    }
    return true;
}

} // namespace dflash::common
