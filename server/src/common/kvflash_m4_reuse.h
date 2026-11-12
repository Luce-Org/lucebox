// Mode 4 on a turn that continues the same human question (a coding agent appending tool results): pure decision
// for running the turn once instead of twice.
//
// The full path (first pass on the recency pool, exact scorer, rewind, second pass) runs whenever the question is
// new or the conversation is not the one last served. On a continuing turn:
//   Reuse     every recall candidate (chunk left of the tail window) already has an exact score from the last
//             scoring, and none of those chunks was rewritten since: pick again from those scores, no scorer.
//   Prescore  a candidate has no score yet: run the exact scorer before the pass, with the question's stored queries
//             and the tail queries the previous request left, then pick. No rewind either.
// No candidate is ever ranked by a score the scorer did not compute for it (mode 5's reuse padded unseen chunks with
// the minimum, which kept everything newer than the last drafter call out of the recall).

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace luce::common {

// What the last mode-4 request left behind.
struct KvFlashM4Last {
    bool                 valid = false;
    int                  uq_hi = -1;      // key of the latest human question: {end position, hash of ids[0, end)}
    uint64_t             uq_hash = 0;
    std::vector<int32_t> ids;             // that request's prompt
    std::vector<float>   scores;          // exact scores of its first scores.size() chunks, as the scorer returned them
    std::vector<float>   q_tail;          // its last prompt queries (selector v2's tail queries)
};

// Tokens at the end of the last prompt the next one may render differently (the generation prompt becomes the turn).
constexpr int kKvFlashM4TailSlack = 64;

enum class KvFlashM4Path { Full, Reuse, Prescore };

struct KvFlashM4Ask {
    int      uq_hi = -1;
    uint64_t uq_hash = 0;
    bool     uq_stored = false;   // the question's query rows are in the store
    int      kv_offset = 0;       // first position this request prefills
    int      chunk_tokens = 0;
    int      first_window = 0;    // first chunk of the tail window when the pick is made at kv_offset
};

inline KvFlashM4Path kvflash_m4_path(const KvFlashM4Last & last, const KvFlashM4Ask & a, const std::vector<int32_t> & ids) {
    if (!last.valid || a.uq_hi <= 0 || !a.uq_stored || a.chunk_tokens <= 0 || last.uq_hi != a.uq_hi || last.uq_hash != a.uq_hash ||
        ids.size() < last.ids.size())
        return KvFlashM4Path::Full;
    const int64_t lcp = std::mismatch(last.ids.begin(), last.ids.end(), ids.begin()).first - last.ids.begin();
    if (lcp < (int64_t)last.ids.size() - kKvFlashM4TailSlack) return KvFlashM4Path::Full;
    const int64_t scored = (int64_t)last.scores.size(), scored_tokens = scored * a.chunk_tokens;
    if (scored > 0 && a.first_window <= scored && a.kv_offset >= scored_tokens && lcp >= scored_tokens) return KvFlashM4Path::Reuse;
    return last.q_tail.empty() ? KvFlashM4Path::Full : KvFlashM4Path::Prescore;
}

}  // namespace luce::common
