#pragma once

#include <cstddef>

namespace dflash::common {

// The historical 16K target graph holds up to eight independent recurrent
// segments. Packed mixed prefill adds one more segment for compact decode:
// on Qwen3.6-27B, eight prefills build 15,061 nodes, while adding decode builds
// 16,741 and overflows the old limit. Each additional segment contributes
// 1,680 nodes (35 nodes in each of 48 DeltaNet layers), so reserve 2K per
// segment beyond the proven eight-segment baseline. This keeps today's common
// shapes at the old capacity while scaling linearly for K=16/32 and beyond.
inline size_t qwen35_target_graph_capacity(
        int n_prefill_segments, bool compact_decode) {
    constexpr size_t kBaseCapacity = 16384;
    constexpr int kBaseRecurrentSegments = 8;
    constexpr size_t kGrowthPerSegment = 2048;

    const int prefill_segments = n_prefill_segments > 0
        ? n_prefill_segments
        : 0;
    const int recurrent_segments =
        prefill_segments + (compact_decode ? 1 : 0);
    if (recurrent_segments <= kBaseRecurrentSegments) {
        return kBaseCapacity;
    }
    return kBaseCapacity +
        (size_t)(recurrent_segments - kBaseRecurrentSegments) *
            kGrowthPerSegment;
}

} // namespace dflash::common
