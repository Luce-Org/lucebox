// LUCE_KVFLASH_PREFILL_RESELECT: pure helpers for reselecting the kvflash pool
// before the final prefill batch (no ggml, no GPU).

#pragma once

#include "kvflash_block_scores.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace luce::common {

// LUCE_KVFLASH_PREFILL_RESELECT: 0 off, 4 reselect by exact attention mass, 5 by the drafter's scores. Both need the
// qk policy and the scorer that matches them; any other value (the retired modes 1-3 included) is off. A mismatch
// is reported in *why (cleared first).
inline int kvflash_reselect_mode(int env_mode, bool qk_policy, const char * scorer, std::string * why) {
    if (why) why->clear();
    if (!qk_policy || (env_mode != 4 && env_mode != 5)) return 0;
    const char * want = env_mode == 4 ? "exact" : "pflash2";
    if (scorer && std::string(scorer) == want) return env_mode;
    if (why) *why = "LUCE_KVFLASH_PREFILL_RESELECT=" + std::to_string(env_mode) + " needs LUCE_KVFLASH_RESELECT_SCORER=" + want;
    return 0;
}

// Which recall features stay on. Recall (mode 4/5) and the user-question queries need the per-token query capture
// (TargetCache::q_cap_all). That tensor has no split in the tensor-parallel layout, so with tp_degree > 1 both are
// off; the pool, the pin and the rest of KVFlash are untouched. `why` is empty when nothing the request asked for was turned off.
struct KvflashFeatureAsk { int reselect_mode = 0; int uq_queries = 0; };
struct KvflashFeaturePlan { int reselect_mode = 0; int uq_queries = 0; std::string why; };
inline KvflashFeaturePlan kvflash_recall_features(int tp_degree, const KvflashFeatureAsk & req) {
    KvflashFeaturePlan plan{req.reselect_mode, req.uq_queries, ""};
    const bool asked = req.reselect_mode != 0 || req.uq_queries > 0;
    if (tp_degree > 1) {
        plan.reselect_mode = plan.uq_queries = 0;
        if (asked) plan.why = "tensor parallelism has no split for the per-token query capture";
    }
    return plan;
}

// Position of the last `<|im_start|>` immediately followed by `user`, or -1.
inline int kvflash_last_user_turn(const std::vector<int32_t> & prompt, int32_t im_start, int32_t user) {
    if (im_start < 0 || user < 0) return -1;
    for (int i = (int)prompt.size() - 2; i >= 0; i--)
        if (prompt[(size_t)i] == im_start && prompt[(size_t)i + 1] == user) return i;
    return -1;
}

// Mode 4 ("rewind the turn"): R4 = start of the last user turn of this
// request's tokens (absolute = kv_offset + index), or the prefill start when
// the tokens hold none. Never below kv_offset.
struct KvFlashMode4Plan {
    int  r = -1;
    bool run = false;
};
// The longest turn a recall replays: the pool minus the pins, the recall and two chunks of room (kvflash_recall_gate's
// PoolTooSmall bound), one of them the turn's partial chunk. 0: no room for a replay (uncapped: the gate refuses).
inline int kvflash_max_turn_tokens(int chunk_tokens, int n_pool_blocks, int pin_reserve_chunks, int recall_chunks) {
    const int room = n_pool_blocks - pin_reserve_chunks - recall_chunks - 2;
    return chunk_tokens > 0 && room > 1 ? (room - 1) * chunk_tokens : 0;
}

// max_turn_tokens > 0: a turn longer than that rewinds to its tail only. Uncapped, a cold single turn (R in chunk 0)
// or a turn longer than the pool skipped the recall and ran LRU only, dropping whatever the pool could not hold.
inline KvFlashMode4Plan kvflash_mode4_plan(const std::vector<int32_t> & tokens, int kv_offset,
                                           int32_t im_start, int32_t user, int max_turn_tokens = 0) {
    KvFlashMode4Plan p;
    if (tokens.empty()) return p;
    const int idx = kvflash_last_user_turn(tokens, im_start, user);
    p.r = kv_offset + (idx > 0 ? idx : 0);
    const int end = kv_offset + (int)tokens.size();
    if (max_turn_tokens > 0 && end - p.r > max_turn_tokens) p.r = end - max_turn_tokens;
    p.run = true;
    return p;
}

// Chunks per recall block (and the sink block recall never takes) unless LUCE_KVFLASH_SELECT_UNIT says otherwise.
constexpr int kKvFlashRecallUnit = 10;

// Recall and compaction are active only when the pool is full: the whole
// prompt plus the room reserved for what follows must NOT fit.
inline bool kvflash_pool_fits(int end, int chunk_tokens, int reserve_chunks, int n_pool_blocks) {
    if (chunk_tokens <= 0) return false;
    return (end + chunk_tokens - 1) / chunk_tokens + reserve_chunks <= n_pool_blocks;
}

// The score the pager's eviction hook reads for chunk `c`. The ledger can cover the whole prompt (mode 5 scores it
// before the 27B runs), but only the first `committed_chunks` are written; the others belong to the batch being
// allocated and must never be picked as victims, so they rank as pinned.
inline float kvflash_hook_score(const std::vector<float> & scores, int committed_chunks, int c) {
    return c < std::min((int)scores.size(), committed_chunks) ? scores[(size_t)c] : 1e30f;
}

// Why a request's recall (mode 4 rewind or mode 5 drafter pick) does not run.
enum class KvFlashRecallSkip { None, NoSealedChunk, Fits, PoolTooSmall };
struct KvFlashRecallGate { KvFlashRecallSkip skip = KvFlashRecallSkip::None; int reserve = 0; };

// `r` is the absolute start of the last user turn, `end` the prompt end. Recall needs a sealed chunk before the turn,
// a prompt that does not already fit the pool (the turn counted once: chunks(r) + the turn's reserve), and a pool that
// holds the preamble pins, the recall and the turn's room beside at least one tail and one victim chunk.
inline KvFlashRecallGate kvflash_recall_gate(bool turn_found, int r, int end, int chunk_tokens, int n_pool_blocks,
                                             int pin_reserve_chunks, int recall_chunks) {
    KvFlashRecallGate g;
    if (!turn_found || chunk_tokens <= 0 || r / chunk_tokens == 0) { g.skip = KvFlashRecallSkip::NoSealedChunk; return g; }
    g.reserve = (end - r + chunk_tokens - 1) / chunk_tokens + 1;
    if (kvflash_pool_fits(r, chunk_tokens, g.reserve, n_pool_blocks)) g.skip = KvFlashRecallSkip::Fits;
    else if (pin_reserve_chunks + recall_chunks + g.reserve + 2 > n_pool_blocks) g.skip = KvFlashRecallSkip::PoolTooSmall;
    return g;
}

// A GPU-scorer refusal that says nothing about the scorer (nothing sealed yet, a transient free-VRAM shortfall) must
// not send every later request to the CPU scorer.
inline bool kvflash_gpu_failure_latches(const std::string & why) {
    return why != "no sealed chunks" && why.rfind("needs ", 0) != 0;
}

// Victim tier (paged-out chunks kept on the device): only mode 5 recalls enough to pay for it.
inline int kvflash_victim_blocks_default(int reselect_mode, int pool_blocks) { return reselect_mode == 5 ? 2 * pool_blocks : 0; }
// Device bytes per pool token the auto pool budget must reserve: the pool plus its victim tier.
inline int kvflash_pool_density_factor(int reselect_mode) { return 1 + kvflash_victim_blocks_default(reselect_mode, 1); }

// Sorted copy of `cuts` with r inserted; unchanged if r < 0 or already present.
inline std::vector<int> kvflash_cuts_with(const std::vector<int> & cuts, int r) {
    std::vector<int> out = cuts;
    if (r >= 0 && std::find(out.begin(), out.end(), r) == out.end()) out.push_back(r);
    std::sort(out.begin(), out.end());
    return out;
}

// R-LRU recall: block b = chunks [cpb*b, cpb*b + cpb); its score is the sum of
// its chunk scores. Eligible blocks lie entirely in [first_chunk, first_window_chunk)
// and inside chunk_scores; the default first_chunk keeps v1's 10-chunk sink block
// (chunk 0 and its neighbours) out at every block size. Returns the top-k block
// ids, best first, ties newer-first.
inline std::vector<int> kvflash_recall_pick(const std::vector<float> & chunk_scores,
                                            int first_window_chunk, int k,
                                            int chunks_per_block = kKvFlashRecallUnit, int first_chunk = kKvFlashRecallUnit) {
    std::vector<int> ids;
    if (k <= 0 || chunks_per_block <= 0) return ids;
    const std::vector<double> bs = kvflash_block_scores(chunk_scores, chunks_per_block, std::max(first_window_chunk, 0));
    for (int b = std::max(1, (first_chunk + chunks_per_block - 1) / chunks_per_block); b < (int)bs.size(); b++) ids.push_back(b);
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) {
        return bs[(size_t)a] != bs[(size_t)b] ? bs[(size_t)a] > bs[(size_t)b] : a > b;
    });
    if ((int)ids.size() > k) ids.resize((size_t)k);
    return ids;
}

// ---- LUCE_KVFLASH_PREAMBLE_ROWS: pin the system+tools preamble outside the history budget ----

// R (tokens) from the env; 0 = off (unset, non-numeric or negative).
inline int kvflash_preamble_rows_from_env() {
    const char * e = std::getenv("LUCE_KVFLASH_PREAMBLE_ROWS");
    if (!e || !*e) return 0;
    char * end = nullptr;
    const long v = std::strtol(e, &end, 10);
    return (*end || v <= 0 || v > 1L << 30) ? 0 : (int)v;
}

// R rounded up to a whole number of chunks (0 when off or the chunk size is unusable).
inline int kvflash_preamble_reserve_tokens(int rows, int chunk_tokens) {
    if (rows <= 0 || chunk_tokens <= 0) return 0;
    return (rows + chunk_tokens - 1) / chunk_tokens * chunk_tokens;
}

// The preamble reserve inside a pool of `pool_tokens`: `rows` rounded up to a chunk, taken out of the pool, which
// never grows (every decode step attends the whole pool, so --kvflash N sets the decode cost). 0 = pin off: no rows
// asked, KVFlash off, an unusable chunk size, or a reserve that leaves no history row (reserve >= pool).
inline int kvflash_preamble_reserve_in_pool(int pool_tokens, int rows, int chunk_tokens) {
    const int reserve = kvflash_preamble_reserve_tokens(rows, chunk_tokens);
    return pool_tokens <= 0 || reserve >= pool_tokens ? 0 : reserve;
}

// End of the preamble: when the prompt starts with `<|im_start|>system`, the index of the next
// `<|im_start|>` (the Qwen template puts the tool schemas inside the system block, so this is
// system + tools); otherwise 0 (no system block, an unknown id, or no turn after it).
inline int kvflash_preamble_end(const std::vector<int32_t> & ids, int32_t im_start, int32_t system_role) {
    if (im_start < 0 || system_role < 0 || ids.size() < 2 || ids[0] != im_start || ids[1] != system_role) return 0;
    for (size_t i = 2; i < ids.size(); i++)
        if (ids[i] == im_start) return (int)i;
    return 0;
}

// Why the preamble pin must stay off for the process ("" = ids present, on).
inline std::string kvflash_preamble_gate(int32_t im_start, int32_t system_role) {
    if (im_start < 0) return "<|im_start|> id missing";
    if (system_role < 0) return "system id missing";
    return "";
}

// Rows to pin: the preamble, never more than the reserve.
inline int kvflash_preamble_pin_rows(int preamble_end, int reserve_rows) {
    return preamble_end <= 0 || reserve_rows <= 0 ? 0 : std::min(preamble_end, reserve_rows);
}

// Chunks the pin covers (an upper bound on what the pager accepts).
inline int kvflash_pin_chunks(int preamble_end, int reserve_rows, int chunk_tokens) {
    const int rows = kvflash_preamble_pin_rows(preamble_end, reserve_rows);
    return rows <= 0 || chunk_tokens <= 0 ? 0 : (rows + chunk_tokens - 1) / chunk_tokens;
}

// Pins the preamble chunks of `pager` (anything with pin_range / is_pinned / chunk_tokens); returns
// the number of chunks actually pinned (the pager refuses a span that would deadlock eviction).
template <class Pager>
inline int kvflash_pin_preamble(Pager & pager, int preamble_end, int reserve_rows) {
    const int rows = kvflash_preamble_pin_rows(preamble_end, reserve_rows);
    const int ct = pager.chunk_tokens();
    if (rows <= 0 || ct <= 0) return 0;
    pager.pin_range(0, rows);
    int n = 0;
    for (int c = 0; c < (rows + ct - 1) / ct; c++) n += pager.is_pinned(c) ? 1 : 0;
    return n;
}

// Recall eligibility floor (chunks): v1's 10-chunk sink block, or past the pinned preamble.
inline int kvflash_recall_first_chunk(int pinned_chunks) { return std::max(kKvFlashRecallUnit, pinned_chunks); }

// Chunks of exact tail window next to `recall_chunks` of recalled blocks and `reserve` chunks of room.
// `pin_reserve_chunks` (the preamble reserve, rows of the pool held for the pins) is taken off the
// pool first, so the history budget is the pool minus the reserve.
inline int kvflash_recall_tail_chunks(int n_pool_blocks, int recall_chunks, int reserve, int pin_reserve_chunks) {
    return std::max(0, n_pool_blocks - pin_reserve_chunks - 1 - recall_chunks - reserve - 1);
}

// ---- recall budget ----
//
// The recall step of kvflash_prefill_apply, pure: how many chunks of exact tail, which blocks are picked.
// `pin_reserve_chunks` is the part of the pool LUCE_KVFLASH_PREAMBLE_ROWS reserves for the pins, off the history budget (0 when off);
// `pinned_chunks` are the leading chunks excluded from the candidates.
struct KvFlashRecallIn {
    int  n_pool_blocks = 0;      // the pool, with the preamble reserve
    int  cur_chunk = 0;
    int  recall_chunks = 0;      // chunks the recall asks for (10 x RECALL_BLOCKS)
    int  unit = kKvFlashRecallUnit;   // chunks per picked block
    int  reserve = 0;            // pool room kept for the rest of the prefill
    int  pin_reserve_chunks = 0;
    int  pinned_chunks = 0;
};
struct KvFlashRecallPlan {
    int tail = 0;
    int k_units = 0;
    int first_chunk = kKvFlashRecallUnit;
    int first_window = 0;
    std::vector<int> picked;
};
// First chunk of the exact tail window: chunks left of it are the recall candidates.
inline int kvflash_recall_first_window(const KvFlashRecallIn & in) {
    return in.cur_chunk - kvflash_recall_tail_chunks(in.n_pool_blocks, in.recall_chunks, in.reserve, in.pin_reserve_chunks);
}
inline KvFlashRecallPlan kvflash_recall_plan(const std::vector<float> & scores, const KvFlashRecallIn & in) {
    KvFlashRecallPlan r;
    r.k_units = in.unit > 0 ? in.recall_chunks / in.unit : 0;
    r.first_chunk = kvflash_recall_first_chunk(in.pinned_chunks);
    r.first_window = kvflash_recall_first_window(in);
    r.tail = in.cur_chunk - r.first_window;
    r.picked = kvflash_recall_pick(scores, r.first_window, r.k_units, in.unit, r.first_chunk);
    return r;
}

}  // namespace luce::common
