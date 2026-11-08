// LUCE_KVFLASH_M5_RESCORE_TOKENS: pure decision for skipping the mode-5 drafter scoring on a turn whose
// latest user question has not changed (coding-agent turns that only append tool results).

#pragma once

#include "fnv.h"

#include <cstdint>
#include <vector>

namespace luce::common {

// What the last drafter scoring of this conversation saw.
struct KvFlashM5Last {
    bool     valid = false;
    uint64_t query_key = 0;     // hash of the latest user message's tokens
    int      end = 0;           // prompt length that was scored
    uint64_t prefix_hash = 0;   // fnv over ids[0, end)
};

// Hash of ids[lo, hi): the key of the latest user question.
inline uint64_t kvflash_m5_query_key(const std::vector<int32_t> & ids, int lo, int hi) {
    (void)ids; (void)lo; (void)hi;
    return 0;
}

// Score unless the last scoring still stands: same question, the prompt extends the scored prompt
// (prefix_hash_at_last_end = fnv over this prompt's ids[0, last.end)), and fewer than rescore_tokens
// new tokens. rescore_tokens <= 0 always scores.
inline bool kvflash_m5_should_score(const KvFlashM5Last & last, uint64_t query_key, int end_tokens,
                                    uint64_t prefix_hash_at_last_end, int rescore_tokens) {
    (void)last; (void)query_key; (void)end_tokens; (void)prefix_hash_at_last_end; (void)rescore_tokens;
    return true;
}

}  // namespace luce::common
