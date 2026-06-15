// freeze_history — pure hash helper for FlowKV freeze-history feature.

#include "server/freeze_history.h"
#include "server/prefix_cache.h"  // hash_prefix

namespace dflash::common {

PrefixHash frozen_block_key(const int32_t * ids, int begin, int end) {
    if (begin >= end) { PrefixHash h{}; return h; }
    return hash_prefix(ids + begin, end - begin);
}

}  // namespace dflash::common
