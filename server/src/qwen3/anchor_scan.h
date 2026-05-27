// N-gram anchor scan: mark chunks forced by token-match between a query pool
// and the body of an ids sequence.  Pure CPU, no GPU, no model required.
#pragma once

#include <climits>
#include <cstdint>
#include <vector>

namespace dflash::qwen3 {

struct AnchorScanCfg {
    int chunk_size;
    int anchor_radius;
    int max_anchor_hits;
    int ngram = 4;
    int rare_token_max_freq = 8;        // tokens appearing <= this many times in body count as rare
    int cascade_min_anchor_count = 0;   // skip cascade if pass-1 forced >= this many chunks (0 = always cascade)
    int max_forced_count = INT_MAX;     // hard cap on total forced chunks
};

// Marks chunks forced by ngram-matches between query_pool and ids[0..body_end).
// `forced` is in-out; new hits are OR-merged.  Idempotent.
void scan_and_force(
    const std::vector<int32_t>& ids,
    int body_end,
    const std::vector<int32_t>& query_pool,
    const AnchorScanCfg& cfg,
    std::vector<uint8_t>& forced
);

// Transitive variant: expands the query pool with tokens from newly-forced
// chunks and re-runs scan_and_force until a fixed point or max_iters reached.
void scan_and_force_transitive(
    const std::vector<int32_t>& ids,
    int body_end,
    const std::vector<int32_t>& initial_query_pool,
    const AnchorScanCfg& cfg,
    int max_iters,
    std::vector<uint8_t>& forced
);

} // namespace dflash::qwen3
