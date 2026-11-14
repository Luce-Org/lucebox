// LUCE_KVFLASH_UQ_QUERIES: the latest human question as extra selector queries. Pure helpers
// (no ggml, no GPU): the human-turn span, the capture plan, the per-session store with its key.

#pragma once

#include "fnv.h"
#include "kvflash_query_capture.h"

#include <algorithm>
#include <utility>
#include <cstdint>
#include <string>
#include <vector>

namespace luce::common {

// Content span [lo, hi) of a user turn: lo is after `<|im_start|> user \n`, hi the `<|im_end|>`.
struct KvFlashHumanTurn {
    int lo = -1, hi = -1;
    bool valid() const { return lo >= 0 && hi > lo; }
};

// The last `<|im_start|>user` turn that is not a tool result. Qwen templates render tool results as
// user turns whose content starts with `<tool_response>`; without that id (tool_response < 0) they cannot
// be told from human turns, so there is no answer (the load gate keeps the feature off in that case too).
// A turn with no closing `<|im_end|>` is skipped.
inline const char * kvflash_chat_ids_missing(int32_t im_start, int32_t user, int32_t im_end, int32_t tool_response) {
    if (im_start < 0) return "<|im_start|> id missing";
    if (user < 0) return "user id missing";
    if (im_end < 0) return "<|im_end|> id missing";
    if (tool_response < 0) return "<tool_response> is not a single token";
    return "";
}

inline KvFlashHumanTurn kvflash_last_human_turn(const std::vector<int32_t> & ids, int32_t im_start, int32_t user,
                                                int32_t newline, int32_t im_end, int32_t tool_response) {
    if (*kvflash_chat_ids_missing(im_start, user, im_end, tool_response)) return {};
    for (int i = (int)ids.size() - 2; i >= 0; i--) {
        if (ids[(size_t)i] != im_start || ids[(size_t)i + 1] != user) continue;
        int lo = i + 2;
        if (newline >= 0 && lo < (int)ids.size() && ids[(size_t)lo] == newline) lo++;
        if (lo < (int)ids.size() && ids[(size_t)lo] == tool_response) continue;
        int hi = lo;
        while (hi < (int)ids.size() && ids[(size_t)hi] != im_end) hi++;
        if (hi < (int)ids.size() && hi > lo) return {lo, hi};
    }
    return {};
}

struct KvFlashUqPlan {
    bool capture = false;   // this prefill covers the whole window: cut at `cut`, capture rows
    int  n = 0;             // rows wanted: min(N, question tokens)
    int  win_lo = 0, cut = 0;
};
inline KvFlashUqPlan kvflash_uq_plan(const KvFlashHumanTurn & t, int n_queries, int kv_offset, int end) {
    KvFlashUqPlan p;
    if (n_queries <= 0 || !t.valid()) return p;
    p.n = std::min(n_queries, t.hi - t.lo);
    p.win_lo = t.hi - p.n;
    p.cut = t.hi;
    p.capture = t.hi <= end && kv_offset <= p.win_lo;
    return p;
}

// A batch [batch_start, batch_start + batch_tokens) captured its last `captured_rows` queries. Which of
// them fall in the window [win_lo, win_hi): the first row index and the count.
struct KvFlashUqTake { int first_row = 0, count = 0; };
inline KvFlashUqTake kvflash_uq_batch_take(int batch_start, int batch_tokens, int captured_rows, int win_lo, int win_hi) {
    KvFlashUqTake t;
    const int end = batch_start + batch_tokens, rows_lo = end - std::min(captured_rows, batch_tokens);
    const int lo = std::max(win_lo, rows_lo), hi = std::min(win_hi, end);
    if (hi <= lo) return t;
    t.first_row = lo - rows_lo;
    t.count = hi - lo;
    return t;
}

// The captured rows of one session, keyed by {hi, hash of ids[0, hi)}. Not persisted into snapshots.
struct KvFlashUqStore {
    int lo = -1, hi = -1, n = 0;    // the n rows are the queries of positions [hi - n, hi)
    uint64_t hash = 0;
    std::vector<float> rows;        // n * (layers * q heads * head dim), oldest first
    bool empty() const { return n <= 0; }
    void clear() { lo = hi = -1; n = 0; hash = 0; rows.clear(); }
};

inline uint64_t kvflash_uq_hash(const std::vector<int32_t> & ids, int upto) {
    uint64_t h = kFnvBasis;
    for (int i = 0; i < upto && i < (int)ids.size(); i++) h = fnv1a_mix(h, (uint32_t)ids[(size_t)i]);
    return h;
}

// Up to this many sessions keep a store (host memory: n rows * layers * q heads * head dim floats each).
constexpr size_t kKvFlashUqMaxStores = 8;

// LRU of stores, most recent first, keyed by {hi, hash of ids[0, hi)}: the key is the content of the
// prompt prefix, so a store stays valid whatever the pager or history did since.
inline int kvflash_uq_find(const std::vector<KvFlashUqStore> & v, int hi, uint64_t hash) {
    for (size_t i = 0; i < v.size(); i++)
        if (!v[i].empty() && v[i].hi == hi && v[i].hash == hash) return (int)i;
    return -1;
}
// Moves entry i to the front (a hit).
inline void kvflash_uq_touch(std::vector<KvFlashUqStore> & v, int i) {
    if (i > 0 && i < (int)v.size()) std::rotate(v.begin(), v.begin() + i, v.begin() + i + 1);
}
inline void kvflash_uq_erase(std::vector<KvFlashUqStore> & v, int hi, uint64_t hash) {
    const int i = kvflash_uq_find(v, hi, hash);
    if (i >= 0) v.erase(v.begin() + i);
}
// Inserts at the front (replacing an entry with the same key); returns the `hi` of every entry pushed out past `max`.
inline std::vector<int> kvflash_uq_put(std::vector<KvFlashUqStore> & v, KvFlashUqStore && s, size_t max) {
    kvflash_uq_erase(v, s.hi, s.hash);
    v.insert(v.begin(), std::move(s));
    std::vector<int> evicted;
    while (v.size() > max) { evicted.push_back(v.back().hi); v.pop_back(); }
    return evicted;
}

// The forced batch cut: the question's end, whenever it falls strictly inside the prefill (-1 otherwise),
// independent of any store, so identical requests get identical batch schedules.
inline int kvflash_uq_cut(const KvFlashHumanTurn & t, int n_queries, int kv_offset, int end) {
    return n_queries > 0 && t.valid() && t.hi > kv_offset && t.hi < end ? t.hi : -1;
}
// A mode-4 rewind to r replays the window from the new pool when r reaches back to its start.
inline bool kvflash_uq_rewind_recaptures(const KvFlashUqPlan & p, int r) { return p.n > 0 && r <= p.win_lo; }
// kvflash_mode4_plan's tail cap can put R inside the window. A replay from there would recompute the window's tail
// under the new pool while the stored rows keep the old pool's, so R moves to the window's end: the window is not replayed.
inline int kvflash_uq_snap_rewind(const KvFlashUqPlan & p, int r) { return p.n > 0 && r > p.win_lo && r < p.cut ? p.cut : r; }

// Append the stored rows to the scorer's query list (no-op for an empty store).
inline void kvflash_uq_append_queries(std::vector<const float *> & qs, const KvFlashUqStore & s, size_t qsize) {
    if (s.empty() || qsize == 0 || s.rows.size() != (size_t)s.n * qsize) return;
    for (int i = 0; i < s.n; i++) qs.push_back(s.rows.data() + (size_t)i * qsize);
}

// Why the feature must stay off for the process ("" = ids present, on). Tool results can only be told
// from human turns by the single <tool_response> token, so without it the feature fails closed.
inline std::string kvflash_uq_gate(int32_t im_start, int32_t user, int32_t im_end, int32_t tool_response) {
    return kvflash_chat_ids_missing(im_start, user, im_end, tool_response);
}

}  // namespace luce::common
