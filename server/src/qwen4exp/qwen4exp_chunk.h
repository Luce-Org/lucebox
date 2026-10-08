#pragma once

#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <limits>

namespace luce::common {

// The gfx1151 MoE route sort serves at most 32768 prompt rows.
constexpr int kQwen4ExpMaxChunk = 32768;

// Largest 256-row tile multiple within a measured workspace budget, capped at
// kQwen4ExpMaxChunk. Above 512, the dense/QSA peak envelope grows with T. Small
// contexts / low-memory fallbacks also try 256, 128, ... 1. Keep 10% of
// available memory for runtime, driver and OS growth.
template<class Measure>
int qwen4exp_fit_chunk(int max_ctx, size_t available, size_t fixed, Measure measure,
                     size_t * snapshot_budget = nullptr, int floor_rows = 4096) {
    const size_t budget = available - available / 10;
    if (snapshot_budget) {
        const size_t requested = *snapshot_budget;
        *snapshot_budget = 0;
        // Prefill speed first: reserve floor_rows (4096; split mode's prompt
        // pipeline needs 8192), or the fastest smaller chunk that fits.
        // Snapshots get only the remaining safe budget.
        const int floor = qwen4exp_fit_chunk(std::min(max_ctx, floor_rows), available, fixed, measure);
        if (!floor) return 0;
        *snapshot_budget = std::min(requested, budget - fixed - measure(floor));
        fixed += *snapshot_budget;
    }
    if (max_ctx <= 0 || fixed >= budget) return 0;
    int best = 0, lo = 2, hi = std::min(max_ctx, kQwen4ExpMaxChunk) / 256;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (measure(mid * 256) <= budget - fixed) { best = mid * 256; lo = mid + 1; }
        else hi = mid - 1;
    }
    if (best) return best;
    int chunk = 256;
    while (chunk > max_ctx) chunk /= 2;
    for (; chunk; chunk /= 2) if (measure(chunk) <= budget - fixed) return chunk;
    return 0;
}

struct Qwen4ExpWeights;
struct Qwen4ExpCache;
// Call after weights and resident_slots caches are allocated, before prefill.
// Other slots are reserved arithmetically, without risking trial allocations.
// Optional prefix allowance: remaining memory after headroom, runtime state,
// and a 4096-row chunk, 8192 in split mode (or the fastest smaller chunk if
// that cannot fit).
// The caller must enforce the returned allowance on later snapshot captures.
int qwen4exp_select_chunk(ggml_backend_t backend, const Qwen4ExpWeights & w,
    Qwen4ExpCache & cache, int slots = 1, int resident_slots = 1, size_t * snapshot_budget = nullptr);

}  // namespace luce::common
