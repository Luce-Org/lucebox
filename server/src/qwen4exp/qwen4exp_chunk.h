#pragma once

#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace luce::common {

// The gfx1151 MoE route sort serves at most 32768 prompt rows.
constexpr int kQwen4ExpMaxChunk = 32768;
// Split mode: the generic MoE id helper on the expert device serves at most 16384 rows.
constexpr int kQwen4ExpSplitMaxChunk = 16384;
// Split mode: a prompt chunk pipelines over 2 to 4 streams of about this many rows.
constexpr int kQwen4ExpPipelineStreamRows = 4096;
// Split mode: the smallest chunk that still gives the prompt pipeline two streams.
constexpr int kQwen4ExpSplitChunkFloor = 2 * kQwen4ExpPipelineStreamRows;
// Split mode: the hot experts' share of the target unless LUCE_EXPERT_BUDGET_MB sets it. The rest of the target holds
// the dense weights, the KV cache and the prompt-chunk buffers.
constexpr uint64_t kQwen4ExpHotExpertBudget = 8ull << 30;
// Concurrent serving: a prompt forward covers at most this many rows per step while other slots
// decode, so a long prompt holds the live streams for one granule at a time.
constexpr int kQwen4ExpConcurrentGranule = 2048;

// Largest 256-row tile multiple within a measured workspace budget, capped at
// max_rows. Above 512, the dense/QSA peak envelope grows with T. Small
// contexts / low-memory fallbacks also try 256, 128, ... 1. Keep 10% of
// available memory for runtime, driver and OS growth.
template<class Measure>
int qwen4exp_fit_chunk(int max_ctx, size_t available, size_t fixed, Measure measure,
                     size_t * snapshot_budget = nullptr, int floor_rows = 4096,
                     int max_rows = kQwen4ExpMaxChunk) {
    const size_t budget = available - available / 10;
    floor_rows = std::min(floor_rows, max_rows);
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
    int best = 0, lo = 2, hi = std::min(max_ctx, max_rows) / 256;
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

// --max-ctx auto: the next context to try after a plan at `ctx` lacks `missing` bytes, when one context position
// takes `position_bytes` across every slot. At least one step lower, in multiples of `step`; 0 when none is left.
inline int qwen4exp_shrink_context(int ctx, size_t missing, size_t position_bytes, int step = 4096) {
    const int64_t fewer = position_bytes ? (int64_t) std::min<size_t>((missing + position_bytes - 1) / position_bytes,
                                                                      (size_t) ctx)
                                         : ctx;
    const int64_t next = std::min<int64_t>((int64_t) ctx - step, ((int64_t) ctx - fewer) / step * step);
    return next >= step ? (int) next : 0;
}

// --max-ctx auto: the search for the largest context that fits.
struct Qwen4ExpContextFit {
    int fits = 0, misses = 0;        // the largest context that fit and the smallest that did not (0: none yet)
    size_t spare = 0, missing = 0;   // the bytes the fit left over and the miss lacked

    // The next context to plan; 0 once settled (the answer is `fits`, none when it is 0). Until a context fits, a
    // miss shrinks by its shortfall, which overshoots because the graph workspaces shrink with the context too.
    // Then interpolation between the fit and the miss, kept strictly between them, settles to a `step` multiple.
    int next(size_t position_bytes, int step = 4096) const {
        if (!fits) return qwen4exp_shrink_context(misses, missing, position_bytes, step);
        if (!misses || misses - fits <= step) return 0;
        const double t = (double) spare / ((double) spare + (double) missing);
        const int64_t guess = (int64_t) (fits + t * (misses - fits)) / step * step;
        return (int) std::clamp<int64_t>(guess, (int64_t) fits + step, (int64_t) misses - step);
    }
};

// What a chunk plan measured, for fitting the context to the device.
struct Qwen4ExpChunkPlan {
    size_t available = 0;       // device memory free beside the weights and the resident cache
    size_t required = 0;        // the other slots' caches, resident graphs, a floor-sized chunk and the headroom
    size_t position_bytes = 0;  // one context position across every slot's cache
};

struct Qwen4ExpWeights;
struct Qwen4ExpCache;
// Call after weights and resident_slots caches are allocated, before prefill.
// Other slots are reserved arithmetically, without risking trial allocations.
// Optional prefix allowance: remaining memory after headroom, runtime state,
// and a 4096-row chunk, 8192 in split mode (or the fastest smaller chunk if
// that cannot fit).
// The caller must enforce the returned allowance on later snapshot captures.
int qwen4exp_select_chunk(ggml_backend_t backend, const Qwen4ExpWeights & w,
    Qwen4ExpCache & cache, int slots = 1, int resident_slots = 1, size_t * snapshot_budget = nullptr,
    Qwen4ExpChunkPlan * plan = nullptr);

}  // namespace luce::common
