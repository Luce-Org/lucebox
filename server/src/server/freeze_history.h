// freeze_history — FlowKV: hash helper for per-message compression cache keying.
// Partitions turns into verbatim prefix (system), frozen aged region, and hot tail.
// Pure functions: no IO, no globals, no CUDA deps.

#pragma once

#include "server/prefix_cache.h"  // PrefixHash

#include <cstdint>
#include <vector>

namespace dflash::common {

// Stable content-hash of token slice [begin, end); zeroed hash on empty slice.
PrefixHash frozen_block_key(const int32_t * ids, int begin, int end);

}  // namespace dflash::common
