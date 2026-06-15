// KvFlashScorer — pluggable chunk-relevance policy for KvFlashPager.
//
// The pager is policy-agnostic: with no scorer attached it evicts LRU and
// never recalls. A scorer upgrades eviction and reselect() to relevance-
// driven residency (FlashMemory's Memory Indexer role). This interface is
// deliberately dependency-free so the pager runs without pflash, without a
// drafter, and without any model beyond the target.
//
// Implementations:
//   - (none)            pure LRU + recency, zero dependencies
//   - KvFlashDrafterScorer   qwen3/qwen3_kvflash_scorer.h — pflash drafter tail
//                       attention (shared with pflash compression)

#pragma once

#include <cstdint>
#include <vector>

namespace dflash::common {

struct KvFlashScorer {
    virtual ~KvFlashScorer() = default;

    // Fill out[c] with a relevance score (higher = keep resident) for each
    // chunk_tokens-sized chunk of `ids` (the full token history: prompt +
    // generated). Returns false on failure; the caller skips reselect for
    // that round and the pager keeps its LRU behavior.
    virtual bool score_chunks(const std::vector<int32_t> & ids,
                              int chunk_tokens,
                              std::vector<float> & out) = 0;
};

} // namespace dflash::common
