// Per-token query capture for selector v2 and the user-question queries.
//
// The prefill graph keeps the last kQueryCaptureTokens tokens' queries of every batch
// (TargetCache::q_cap_all), so the graph, the memory estimate and the backend agree on one number.

#pragma once

#include "kvflash_pool_flag.h"

#include <algorithm>
#include <cstdlib>

namespace luce::common {

// Per-token queries the graph captures per batch.
constexpr int kQueryCaptureTokens = 64;

// Rows of the verify-query stage (TargetCache::q_stage): the spec verify pass
// stages the query of each of its rows so q_cap can take the last accepted one.
// Only a KVFlash process (LUCE_KVFLASH names a pool) allocates it.
inline int kvflash_q_stage_rows(int max_verify_tokens) {
    return kvflash_pool_requested(std::getenv("LUCE_KVFLASH")) ? std::max(max_verify_tokens, 0) : 0;
}

} // namespace luce::common
