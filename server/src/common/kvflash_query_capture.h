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

// Only a KVFlash process (LUCE_KVFLASH names a pool) keeps the per-token queries: a pooled prefill
// scores with them (selector v2, the user-question queries). Without a pool nothing is allocated.
// The gate reads the env only, not the resolved recall plan: every process that builds a cache
// (create_target_cache_partial, including layer-split shards) reserves the rows. Under tp_degree > 1
// kvflash_recall_features turns recall off, so that reservation is unused: a deliberate per-process
// cost of 64 x head_dim x n_head x n_full_attn x 4 B (about 25 MB at 256 x 24 x 16) in exchange for
// not threading the plan through every cache builder and the VRAM estimate.
inline bool kvflash_pool_process() {
    return kvflash_pool_requested(std::getenv("LUCE_KVFLASH"));
}
inline int kvflash_q_cap_all_rows() { return kvflash_pool_process() ? kQueryCaptureTokens : 0; }

// LUCE_KVFLASH_UQ_QUERIES=N: the latest human question's last N queries join the selector's.
// 0 when unset or invalid, at most the capture window.
inline int kvflash_uq_queries() {
    const char * e = std::getenv("LUCE_KVFLASH_UQ_QUERIES");
    if (!e || !*e) return 0;
    char * end = nullptr;
    const long v = std::strtol(e, &end, 10);
    return (*end || v <= 0) ? 0 : (int)std::min<long>(v, kQueryCaptureTokens);
}

// Rows of the verify-query stage (TargetCache::q_stage): the spec verify pass
// stages the query of each of its rows so q_cap can take the last accepted one.
// Only a KVFlash process (LUCE_KVFLASH names a pool) allocates it.
inline int kvflash_q_stage_rows(int max_verify_tokens) {
    return kvflash_pool_process() ? std::max(max_verify_tokens, 0) : 0;
}

} // namespace luce::common
