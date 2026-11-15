// LUCE_KVFLASH_UQ_QUERIES: the latest human question as extra selector queries. Pure helpers
// (no ggml, no GPU): the human-turn span, the capture plan, the per-session store with its key.

#pragma once

#include "fnv.h"
#include "kvflash_query_capture.h"

#include <algorithm>
#include <utility>
#include <cstdint>
#include <cstdio>
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

// The slot engine captures per cut: how many rows of the cut [pos, pos + n) are inside the window while the
// plan still captures. No cut spans p.cut (the forced cut), so they are the cut's last rows.
inline int kvflash_uq_cut_rows(const KvFlashUqPlan & p, int pos, int n) {
    return p.capture && pos + n <= p.cut ? std::max(0, pos + n - std::max(pos, p.win_lo)) : 0;
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

// The store holds the rows of the question that ends at hi after the prompt prefix with this hash.
inline bool kvflash_uq_holds(const KvFlashUqStore & s, int hi, uint64_t hash) { return !s.empty() && s.hi == hi && s.hash == hash; }
// Is the store the question rows of this prompt's own ids (a session's history)? The window lies in it, n rows fit the window, the
// rows are whole, and the key is the hash of the ids [0, hi). A slot's store outlives the session that captured it until the next
// one replaces it; a bundle save keeps only rows that pass this.
inline bool kvflash_uq_belongs(const KvFlashUqStore & s, const std::vector<int32_t> & ids) {
    return !s.empty() && s.lo >= 0 && s.lo < s.hi && s.hi <= (int)ids.size() && s.n <= s.hi - s.lo && !s.rows.empty() &&
           s.rows.size() % (size_t)s.n == 0 && s.hash == kvflash_uq_hash(ids, s.hi);
}

// LRU of stores, most recent first, keyed by {hi, hash of ids[0, hi)}: the key is the content of the
// prompt prefix, so a store stays valid whatever the pager or history did since.
inline int kvflash_uq_find(const std::vector<KvFlashUqStore> & v, int hi, uint64_t hash) {
    for (size_t i = 0; i < v.size(); i++)
        if (kvflash_uq_holds(v[i], hi, hash)) return (int)i;
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

// The slot engine keeps one such LRU per slot, never shared: a scorer that read another slot's rows would depend on
// how far that slot's request had run. A side request with another last human question (a regenerate, a title call)
// then leaves the session's own question rows in place. 4 a slot, not 8: a store is 24 MiB on qwen35-27B at 64 rows
// (64 * 16 layers * 24 q heads * 256 * 4 bytes), so 4 slots hold at most 384 MiB.
constexpr size_t kKvFlashUqSlotMaxStores = 4;
using KvFlashUqSlots = std::vector<std::vector<KvFlashUqStore>>;
inline std::vector<int> kvflash_uq_slot_put(KvFlashUqSlots & s, int slot, KvFlashUqStore && st, size_t max = kKvFlashUqSlotMaxStores) {
    return kvflash_uq_put(s[(size_t)slot], std::move(st), max);
}
// A slot holds the stores of several questions: a bundle save keeps every one that belongs to the parked history, most recent
// first, at most `max`. A side question's park then keeps the session's own question rows beside the side question's.
inline std::vector<const KvFlashUqStore *> kvflash_uq_saved(const std::vector<KvFlashUqStore> & v, const std::vector<int32_t> & ids,
                                                            size_t max = kKvFlashUqSlotMaxStores) {
    std::vector<const KvFlashUqStore *> out;
    for (const KvFlashUqStore & s : v)
        if (out.size() < max && kvflash_uq_belongs(s, ids)) out.push_back(&s);
    return out;
}
// A bundle restore: the slot holds the saved stores alone (most recent first, as kvflash_uq_saved gave them), in that LRU order.
inline void kvflash_uq_slot_restore(KvFlashUqSlots & s, int slot, std::vector<KvFlashUqStore> && saved, size_t max = kKvFlashUqSlotMaxStores) {
    s[(size_t)slot].clear();
    for (auto it = saved.rbegin(); it != saved.rend(); ++it) kvflash_uq_slot_put(s, slot, std::move(*it), max);
}
// The slot's store for {hi, hash}, now its most recent; null when the slot holds none.
inline const KvFlashUqStore * kvflash_uq_slot_get(KvFlashUqSlots & s, int slot, int hi, uint64_t hash) {
    std::vector<KvFlashUqStore> & v = s[(size_t)slot];
    const int i = kvflash_uq_find(v, hi, hash);
    if (i < 0) return nullptr;
    kvflash_uq_touch(v, i);
    return &v.front();
}

// The stores are host memory no request reserved: the engine holds a list's bytes on its host ledger. `hold(delta)` charges
// delta bytes (negative: gives them back) and answers false when the budget refuses, with nothing charged.
inline int64_t kvflash_uq_bytes(const KvFlashUqStore & s) { return (int64_t)(s.rows.size() * sizeof(float)); }
inline int64_t kvflash_uq_bytes(const std::vector<KvFlashUqStore> & v) {
    int64_t b = 0;
    for (const KvFlashUqStore & s : v) b += kvflash_uq_bytes(s);
    return b;
}
// Which slots' stores a reclaim keeps: a live request's and a parked session's. `own` (the parked slot a resume is asking for, -1:
// none) is given up when nothing is live: its session's own rows would hold the room it asks for and no retry could change the
// answer (kvflash_evict_for_cold's rule). The rows are derived data: the session resumes without them and a new question recaptures.
inline std::vector<bool> kvflash_uq_keep(std::vector<bool> active, const std::vector<bool> & parked, int own = -1) {
    const bool any_live = std::find(active.begin(), active.end(), true) != active.end();
    for (size_t s = 0; s < active.size(); s++) active[s] = active[s] || (s < parked.size() && parked[s] && (any_live || (int)s != own));
    return active;
}
// The stores are a cache: under host pressure those of the slots nobody owns (keep[slot] false: no live request and no parked
// session) are released. Returns what was given back; nothing reclaimable = 0 stores, the ledger untouched.
struct KvFlashUqReclaimed { int stores = 0; int64_t bytes = 0; };
template <class Hold>
KvFlashUqReclaimed kvflash_uq_reclaim_idle(KvFlashUqSlots & s, const std::vector<bool> & keep, Hold && hold) {
    KvFlashUqReclaimed r;
    for (size_t i = 0; i < s.size(); i++) {
        if (i < keep.size() && keep[i]) continue;
        r.stores += (int)s[i].size();
        r.bytes += kvflash_uq_bytes(s[i]);
        s[i].clear();
    }
    if (r.bytes) hold(-r.bytes);
    return r;
}
// What kvflash_uq_put would change the list's bytes by: the new store, less the same key's old rows and the entries past `max`.
inline int64_t kvflash_uq_put_delta(const std::vector<KvFlashUqStore> & v, const KvFlashUqStore & s, size_t max) {
    const int same = kvflash_uq_find(v, s.hi, s.hash);
    int64_t after = max > 0 ? kvflash_uq_bytes(s) : 0;
    size_t kept = 1;
    for (size_t i = 0; i < v.size() && kept < max; i++)
        if ((int)i != same) { after += kvflash_uq_bytes(v[i]); kept++; }
    return after - kvflash_uq_bytes(v);
}
// kvflash_uq_put under the ledger. Refused: false, the list and `evicted` are as they were and the new rows are dropped.
template <class Hold>
bool kvflash_uq_put_held(std::vector<KvFlashUqStore> & v, KvFlashUqStore && s, size_t max, Hold && hold, std::vector<int> * evicted = nullptr) {
    if (!hold(kvflash_uq_put_delta(v, s, max))) return false;
    std::vector<int> ev = kvflash_uq_put(v, std::move(s), max);
    if (evicted) *evicted = std::move(ev);
    return true;
}
template <class Hold>
void kvflash_uq_clear_held(std::vector<KvFlashUqStore> & v, Hold && hold) {
    hold(-kvflash_uq_bytes(v));
    v.clear();
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

// LUCE_KVFLASH_UQ_QUERIES: the rows of the latest human question join the scorer's queries, one hit or miss line per
// reselect. `st` is the store holding this request's question; null or unusable, `miss` says why. The store and the
// selector decide alone: a call too large for the GPU scorer runs on the CPU scorer with the same queries.
// Returns the miss reason, nullptr on a hit.
inline const char * kvflash_uq_join(std::vector<const float *> & qs, const KvFlashUqStore * st, const char * miss, size_t qsize, size_t n_v2) {
    const char * why = n_v2 == 0 ? "no-v2-selector"
        : !st || st->empty() || qsize == 0 || st->rows.size() != (size_t)st->n * qsize ? (miss && *miss ? miss : "no-store") : nullptr;
    if (why) {
        std::fprintf(stderr, "[kvflash] uq miss reason=%s\n", why);
        return why;
    }
    for (int i = 0; i < st->n; i++) qs.push_back(st->rows.data() + (size_t)i * qsize);
    std::fprintf(stderr, "[kvflash] uq hit lo=%d hi=%d n=%d\n", st->lo, st->hi, st->n);
    return nullptr;
}

// Why the feature must stay off for the process ("" = ids present, on). Tool results can only be told
// from human turns by the single <tool_response> token, so without it the feature fails closed.
inline std::string kvflash_uq_gate(int32_t im_start, int32_t user, int32_t im_end, int32_t tool_response) {
    return kvflash_chat_ids_missing(im_start, user, im_end, tool_response);
}

}  // namespace luce::common
