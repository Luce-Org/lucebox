// LUCE_KVFLASH_M5_RESCORE_TOKENS: pure decision for skipping the mode-5 drafter scoring on a turn whose
// latest user question has not changed (coding-agent turns that only append tool results).

#pragma once

#include "fnv.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace luce::common {

// What the last drafter scoring of this conversation saw.
struct KvFlashM5Last {
    bool                 valid = false;
    uint64_t             query_key = 0;   // hash of the latest user message's tokens
    std::vector<int32_t> ids;             // the scored prompt
};

// Tokens at the end of the last scored prompt that the next prompt may render differently: the generation prompt
// becomes the assistant turn.
constexpr int kKvFlashM5TailSlack = 64;

// Default LUCE_KVFLASH_M5_RESCORE_TOKENS: re-score after this many new tokens under an unchanged question
// (2026-11-08 screen: 17 drafter calls instead of 129, P1/P2 held).
constexpr int kKvFlashM5RescoreTokens = 16384;

// Hash of ids[lo, hi): the key of the latest user question.
inline uint64_t kvflash_m5_query_key(const std::vector<int32_t> & ids, int lo, int hi) {
    uint64_t h = kFnvBasis;
    for (int i = lo; i < hi && i < (int)ids.size(); i++) h = fnv1a_mix(h, (uint32_t)ids[(size_t)i]);
    return fnv1a_mix(h, (uint64_t)(hi - lo));
}

// Score unless the last scoring still stands: same question, ``ids`` keeps the last scored prompt up to its last
// kKvFlashM5TailSlack tokens, and fewer than rescore_tokens tokens follow the common prefix. rescore_tokens <= 0
// always scores.
inline bool kvflash_m5_should_score(const KvFlashM5Last & last, uint64_t query_key, const std::vector<int32_t> & ids,
                                    int rescore_tokens) {
    if (rescore_tokens <= 0 || !last.valid || last.query_key != query_key || ids.size() < last.ids.size()) return true;
    const int lcp = (int)(std::mismatch(last.ids.begin(), last.ids.end(), ids.begin()).first - last.ids.begin());
    return lcp < (int)last.ids.size() - kKvFlashM5TailSlack || (int)ids.size() - lcp >= rescore_tokens;
}

// The chunk scores a skipped turn applies: the last drafter scores over ``n_chunks`` chunks; chunks it never saw get
// its lowest score (the recency window keeps them, recall never prefers them over a scored chunk).
inline std::vector<float> kvflash_m5_reuse_scores(const std::vector<float> & last, int n_chunks) {
    if (last.empty() || n_chunks <= 0) return {};
    std::vector<float> r(last.begin(), last.begin() + std::min(last.size(), (size_t)n_chunks));
    r.resize((size_t)n_chunks, *std::min_element(last.begin(), last.end()));
    return r;
}

}  // namespace luce::common
