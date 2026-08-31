// Pure host-side MMQ stream-k grid selection (no CUDA types).
// Shared by mmq.cuh and host unit tests.
#pragma once

#include <cstdint>

static inline int mmq_stream_k_nblocks(
        const int ntiles_dst,
        const int nsm,
        const int64_t ncols_x,
        const int iter_k,
        const bool is_nvidia,
        const bool enable_useful_chunk_cap) {
    if (ntiles_dst <= 0 || nsm <= 0 || ncols_x <= 0 || iter_k <= 0) {
        return 1;
    }

    const int64_t tiles_nwaves = (static_cast<int64_t>(ntiles_dst) + nsm - 1) / nsm;
    const int64_t tiles_efficiency_percent =
        100 * static_cast<int64_t>(ntiles_dst) / (static_cast<int64_t>(nsm) * tiles_nwaves);

    // Preserve the pre-existing NVIDIA >=90% pure-tiling escape.
    if (is_nvidia && tiles_efficiency_percent >= 90) {
        return ntiles_dst;
    }

    // Fail closed outside the specifically validated hardware path.
    if (!enable_useful_chunk_cap) {
        return nsm;
    }

    // The device aligns CTA boundaries down to complete iter_k/qk chunks.
    // Keep a partial K tail with the tile's final CTA instead of counting it
    // as another CTA; otherwise the aligned partition creates empty CTAs.
    const int64_t iters_per_tile = ncols_x >= iter_k ? ncols_x / iter_k : 1;
    // We only need the exact product when it is below nsm. Saturate first so
    // neither division nor multiplication can overflow.
    if (iters_per_tile >= nsm) {
        return nsm;
    }
    const int64_t tiles_needed_to_fill = (nsm + iters_per_tile - 1) / iters_per_tile;
    if (static_cast<int64_t>(ntiles_dst) >= tiles_needed_to_fill) {
        return nsm;
    }
    const int64_t total_iters = static_cast<int64_t>(ntiles_dst) * iters_per_tile;
    return total_iters > 0 ? static_cast<int>(total_iters) : 1;
}

static inline bool mmq_stream_k_fixup_needed(const int ntiles_dst, const int stream_k_blocks) {
    return stream_k_blocks > 0 && ntiles_dst % stream_k_blocks != 0;
}
