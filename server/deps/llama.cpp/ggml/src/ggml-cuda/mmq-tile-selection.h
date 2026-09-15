#pragma once

#include <cstdint>
#include <limits>

// Host-side tile selection shared by CUDA/HIP dispatch and its unit tests.
// Hardware/qtype eligibility and automatic-only exclusions stay with the caller.
template <typename Supported, typename SkipAutomatic>
static int ggml_cuda_mmq_select_x(
        int64_t ncols, int max_x, int requested_x,
        Supported supported, SkipAutomatic skip_automatic) {
    if (requested_x > 0 && requested_x <= max_x && supported(requested_x)) {
        return requested_x;
    }

    int best_x = 0;
    int64_t best_tiles = std::numeric_limits<int64_t>::max();
    // Finding a valid candidate must not end the search: a wider tile may
    // need fewer blocks. Accepted explicit/adaptive requests returned above.
    for (int x = 8; x <= max_x && best_tiles > 1; x += 8) {
        if (!supported(x) || skip_automatic(x)) {
            continue;
        }
        const int64_t tiles = ncols / x + (ncols % x != 0);
        if (tiles < best_tiles) {
            best_x = x;
            best_tiles = tiles;
        }
    }
    return best_x;
}
