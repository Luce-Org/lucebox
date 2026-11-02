// Shared prefill helpers for Qwen3.5/3.6.

#pragma once

#include "attn_masks.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

namespace luce::common {

// Qwen prefill chunking. Between restore points chunks keep to the
// kQwen35PrefillUbatch grid. A chunk never ends within kQwen35MinChunkTokens
// of its start, except at the end of the prompt: a chunk of a few tokens in
// the middle of a prompt produced NaN logits on the R9700 (Qwen3.8, a 2-token
// chunk after a restore). A restore away from every restore point (a
// generated-turn checkpoint) cannot reproduce a cold prefill anyway, so its
// first chunk runs kQwen35OffGridLeadTokens before stopping. Both rules depend
// only on where chunks start, so a cold prefill and a restored one still cut
// alike.
inline constexpr int kQwen35PrefillUbatch = 512;
inline constexpr int kQwen35MinChunkTokens = 16;
inline constexpr int kQwen35OffGridLeadTokens = 64;

// Tokens of the prefill chunk at absolute position kv_pos with `remaining`
// prompt tokens left. Every restore point (ascending, absolute) starts a
// chunk; between them chunks end on multiples of `ubatch`, the grid a cold
// prefill uses. A chunk ends no sooner than `min_tokens` in (at most
// `ubatch`), except at the end of the prompt. The result depends only on where the chunk starts, so a
// prefill resumed at a chunk start of a cold prefill cuts the rest alike.
inline int qwen35_prefill_chunk_tokens(int kv_pos, int remaining, int ubatch,
                                       const std::vector<int> & restore_points,
                                       int min_tokens) {
    const int after = kv_pos - 1 + std::max(1, min_tokens);
    int end = kv_pos + std::min(ubatch, remaining);
    const auto next = std::upper_bound(
        restore_points.begin(), restore_points.end(), after);
    if (next != restore_points.end()) end = std::min(end, *next);
    end = std::min(end, (after / ubatch + 1) * ubatch);
    return end - kv_pos;
}

// Prefill tokens between trims of the legacy (non-VMM) device pools; see
// do_prefill. Flash attention converts the quantized K and V of the whole
// context so far to f16 in pool temporaries, so every few chunks of a long
// prefill ask for a little more than the pool's largest cached block. The pool
// allocates a new block (5% look-ahead) and keeps the outgrown ones, so its
// cache grows with the prompt: one cold 125K-token prefill on a 32 GB R9700
// had cached 5.7 GiB when VRAM ran out at about 87K tokens. Trimming every
// 8192 tokens (16 chunks) kept that prefill at 28.1 GB instead of the full
// card, for the same prefill time.
inline constexpr int kQwen35PrefillPoolTrimTokens = 8192;

// LUCE_PREFILL_POOL_TRIM_TOKENS: unset or empty keeps the default interval, 0
// turns the trims off, a positive count sets the interval. Anything else,
// including a count past INT_MAX, keeps the default and clears *valid.
inline int qwen35_prefill_pool_trim_tokens(const char * value, bool * valid) {
    *valid = true;
    if (value == nullptr || value[0] == '\0') return kQwen35PrefillPoolTrimTokens;
    char * end = nullptr;
    // strtol saturates instead of failing: at LONG_MAX, which is INT_MAX on
    // Windows, only errno tells an overflow from a valid count.
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || errno == ERANGE || parsed < 0 || parsed > INT_MAX) {
        *valid = false;
        return kQwen35PrefillPoolTrimTokens;
    }
    return (int)parsed;
}

inline int qwen35_prefill_ubatch(int fallback) {
    const char * value = std::getenv("LUCE_PREFILL_UBATCH");
    return value ? std::max(1, std::atoi(value)) : fallback;
}

// Pooled kvflash prefill ubatch. The pager only needs chunk-aligned batches, so
// round the configured ubatch down to a chunk multiple and clamp to the pool: a
// ubatch larger than the pool would allocate its later chunks by evicting its
// own earlier, not-yet-computed chunks. One chunk is the floor, which wins over
// the pool clamp if the pool itself were smaller than a chunk (the pager's pool
// is always a positive chunk multiple, so that precedence is only a contract).
inline int kvflash_pooled_ubatch(int prefill_ubatch, int chunk_tokens, int pool_tokens) {
    if (chunk_tokens <= 0) return prefill_ubatch;
    int ub = std::max(prefill_ubatch, chunk_tokens);
    ub = (ub / chunk_tokens) * chunk_tokens;
    if (pool_tokens > 0 && ub > pool_tokens) {
        ub = std::max(chunk_tokens, (pool_tokens / chunk_tokens) * chunk_tokens);
    }
    return ub;
}

// KVFlash pooled prefill on an ABSOLUTE grid of `grid` tokens (U = 512, a
// multiple of the 64-token pager chunk). Batches never start off-grid, except
// the first one after an off-grid resume, so a cut cannot depend on where an
// earlier request happened to start and the unaligned batch accounting of the
// old `min(ubatch, remaining)` plan cannot occur.
//
// End of the batch starting at `pos`: the largest grid point within `max_batch`
// of `pos` (so an off-grid start runs to the grid first, and a start just below
// a grid point folds into a wider batch when max_batch >= 2 * grid), capped at
// `end`. Every batch but the first starts on the grid; every batch but the last
// ends on it. Minimum size: no explicit floor is needed. A non-first,
// non-final batch is at least one grid cell (>= 64 tokens, far above the 2
// tokens that gave NaNs on the R9700); the first batch is at least the gap to
// the grid point, which is >= 64 for the chunk-aligned starts KVFlash produces
// (consuming restores are chunk-aligned). The one short batch is the
// end-of-prompt remainder (as in the dense path); the GPU gate covers
// remainders 1..64. If max_batch < grid the grid shrinks to max_batch; a
// max_batch that is not a multiple of the grid is rounded down to one.
//
//
// S9b: `cuts` are extra batch boundaries off the grid (kvflash_turn_cuts): a
// batch that would straddle one ends there, and the batch after it runs to the
// next grid point. The cut set is a pure function of the prompt tokens and is
// applied on every prefill, so a restore at any cut replays exactly the
// batches a cold prefill runs from there on. Cuts at or before `pos` or at or
// after the natural end are no-ops.
inline int kvflash_prefill_batch_end(int pos, int end, int grid, int max_batch,
                                     const std::vector<int> & cuts = {}) {
    const int g = max_batch < grid ? std::max(1, max_batch) : grid;
    const int maxb = (std::max(max_batch, 1) / g) * g;
    int e = std::min(end, ((pos + maxb) / g) * g);
    for (int c : cuts) if (c > pos && c < e) e = c;
    return e;
}

inline std::vector<std::pair<int, int>> kvflash_prefill_batches(int start, int end, int grid,
                                                                int max_batch,
                                                                const std::vector<int> & cuts = {}) {
    std::vector<std::pair<int, int>> out;
    for (int pos = start; pos < end;) {
        const int e = kvflash_prefill_batch_end(pos, end, grid, max_batch, cuts);
        out.emplace_back(pos, e);
        pos = e;
    }
    return out;
}

// Where a pooled prefill takes the boundary snapshot requested at snap_pos: the
// last batch start in (start, snap_pos] that is a multiple of `chunk`, or -1.
// A pooled snapshot is only consumable at a chunk multiple, and a turn cut can
// start a batch off it, so the batch holding snap_pos does not always qualify.
// snap_pos at or past `end` is the end-of-prefill snapshot's job: -1 here.
inline int kvflash_pooled_snapshot_pos(int start, int end, int grid, int max_batch,
                                       const std::vector<int> & cuts, int snap_pos,
                                       int chunk) {
    if (snap_pos <= start || snap_pos >= end || chunk <= 0) return -1;
    int best = -1;
    for (const auto & b : kvflash_prefill_batches(start, end, grid, max_batch, cuts)) {
        if (b.first > snap_pos) break;
        if (b.first > start && b.first % chunk == 0) best = b.first;
    }
    return best;
}

// LUCE_KVFLASH_RESTORE_CONSUME=1 opts in to carrying a pooled KVFlash session
// across turns: pooled prefix snapshots are saved and a restore consumes them
// instead of re-prefilling the prompt. Default OFF.
inline bool kvflash_restore_consume_enabled() {
    const char * v = std::getenv("LUCE_KVFLASH_RESTORE_CONSUME");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

// ─── DeltaNet state checkpoint ladder (S7) ────────────────────────
// Recurrent state cannot be sliced back to an earlier position, so a prefix
// reuse at common-prefix length L needs a state checkpoint at some C <= L.
// Rungs, all multiples of `grid` (= the S6 batch ends): the system/tools head
// (rounded down), every `stride` tokens, and the grid point at or below the
// prompt end. A head < 0 (unknown) leaves that rung out. S9b: `turn_cut` > 0
// (the off-grid turn rung, see kvflash_turn_cut) is added when it is <= end.
inline std::vector<int> kvflash_state_ladder_positions(int head, int end, int grid, int stride,
                                                       int turn_cut = -1) {
    std::vector<int> out;
    if (grid <= 0 || end < grid) return out;
    if (head >= grid) out.push_back(head / grid * grid);
    if (stride > 0) {
        const int s = (stride + grid - 1) / grid * grid;
        for (int p = s; p <= end; p += s) out.push_back(p);
    }
    out.push_back(end / grid * grid);
    if (turn_cut > 0) out.push_back(turn_cut);
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    while (!out.empty() && out.back() > end) out.pop_back();
    return out;
}

// S9b turn cuts: every position just after an `<|im_start|> assistant \n` header,
// ascending, excluding a header that ends the prompt. The turn rung is the last one.
// Pend - k for the last one: the position just after the last `<|im_start|> assistant \n`
// header of `prompt`, i.e. Pend - k, where k is the length of the generation
// suffix (the think block). The next turn's template re-renders the assistant
// turn as `<|im_start|>assistant\n` + content, so its common prefix with this
// prompt ends exactly here. -1 when there is no header, an id is unknown (< 0),
// or the header ends the prompt (nothing after it, k = 0: no use).
inline std::vector<int> kvflash_turn_cuts(const std::vector<int32_t> & prompt, int32_t im_start,
                                          int32_t assistant, int32_t newline) {
    std::vector<int> out;
    if (im_start < 0 || assistant < 0 || newline < 0) return out;
    for (int i = 3; i < (int)prompt.size(); i++) {
        if (prompt[i - 3] == im_start && prompt[i - 2] == assistant && prompt[i - 1] == newline)
            out.push_back(i);
    }
    return out;
}

// The last of them (the generation prompt of the turn), or -1.
inline int kvflash_turn_cut(const std::vector<int32_t> & prompt, int32_t im_start,
                            int32_t assistant, int32_t newline) {
    const std::vector<int> c = kvflash_turn_cuts(prompt, im_start, assistant, newline);
    return c.empty() ? -1 : c.back();
}

// Greatest checkpoint position C <= L, or -1. `positions` need not be sorted.
inline int kvflash_ladder_pick(const std::vector<int> & positions, int L) {
    int best = -1;
    for (int c : positions) if (c <= L && c > best) best = c;
    return best;
}

struct KvflashLadderEntry {
    int      pos;
    uint64_t tick;   // last use / insertion order; larger is newer
};

// Positions to evict (least recently used first) so that
// entries.size() * ckpt_bytes <= budget_bytes. The head checkpoint (pos ==
// head_pos) and the newest entry (greatest tick) are never evicted, so the
// result may leave the store over budget when only those remain.
inline std::vector<int> kvflash_ladder_evict(const std::vector<KvflashLadderEntry> & entries,
                                             size_t budget_bytes, size_t ckpt_bytes,
                                             int head_pos) {
    std::vector<KvflashLadderEntry> cand;
    uint64_t newest = 0;
    for (const auto & e : entries) newest = std::max(newest, e.tick);
    for (const auto & e : entries) {
        if (e.pos != head_pos && e.tick != newest) cand.push_back(e);
    }
    std::sort(cand.begin(), cand.end(),
              [](const KvflashLadderEntry & a, const KvflashLadderEntry & b) { return a.tick < b.tick; });
    std::vector<int> out;
    size_t total = entries.size() * ckpt_bytes;
    for (const auto & e : cand) {
        if (total <= budget_bytes) break;
        out.push_back(e.pos);
        total -= ckpt_bytes;
    }
    return out;
}

// LUCE_KVFLASH_SESSION is the one switch for KVFlash session restore: the DeltaNet
// state ladder, live restore, generation-end checkpoint (S9) and per-turn
// checkpoints (S9b). Default ON (callers also require KVFlash itself to be
// active); LUCE_KVFLASH_SESSION=0 turns all four off. The turn cuts as batch
// boundaries are not gated (they apply to every pooled prefill, like the 512
// grid), so the session-off arm batches like the session-on arm. Pure so it is
// unit-testable.
inline bool kvflash_session_flag(const char * v) {
    return !(v != nullptr && v[0] == '0' && v[1] == '\0');
}

inline bool kvflash_session_enabled() {
    return kvflash_session_flag(std::getenv("LUCE_KVFLASH_SESSION"));
}

// Position of the generation-end checkpoint: cur_pos, where KV and DeltaNet
// state hold exactly history[0:cur_pos] (a final pending token is in the
// history but not in KV). -1 when the history does not cover cur_pos, e.g. a
// decode path that does not record generated ids.
inline int kvflash_gen_ckpt_pos(int cur_pos, size_t history_size) {
    return cur_pos > 0 && history_size >= (size_t)cur_pos ? cur_pos : -1;
}

// A speculative step verified commit_n tokens, so the recurrent state is
// commit_n tokens ahead of the step start. When the emit loop stops early
// (EOS, cancel) the state must be rolled back to the emitted prefix. A budget
// close repairs itself, and the floor path has its own repair.
inline bool spec_early_stop_needs_repair(int commit_n, int emitted, bool budget_close_fired) {
    return emitted < commit_n && !budget_close_fired;
}

// History after a decode phase: history[0:base] followed by the generated ids
// out_tokens[out_entry:]. Rebuilds from `base` rather than appending, so it is
// idempotent over an AR fallback that already pushed some of the same ids.
inline void kvflash_history_set_generated(std::vector<int32_t> & history, size_t base,
                                          const std::vector<int32_t> & out_tokens,
                                          size_t out_entry) {
    history.resize(std::min(base, history.size()));
    if (out_entry < out_tokens.size())
        history.insert(history.end(), out_tokens.begin() + out_entry, out_tokens.end());
}

// Decision for a restore from the live session (S8a).
//   L             longest common prefix of the prompt and the live history,
//                 capped at cur_pos (the history is exact only up to there)
//   C             checkpoint to resume from: greatest rung <= L, or -1. Also
//                 capped at prompt_len-1: the last prompt token must run so
//                 prefill produces the logits decode starts from.
//   replay_from   first prompt index to prefill (C, or 0 without a rung)
//   discard_above rungs strictly above this describe diverged history
struct KvflashLiveRestore {
    int L = 0;
    int C = -1;
    int replay_from = 0;
    int discard_above = 0;
};

inline KvflashLiveRestore kvflash_live_restore_decide(
        const std::vector<int32_t> & prompt, const std::vector<int32_t> & history,
        int cur_pos, const std::vector<int> & ladder_positions) {
    KvflashLiveRestore d;
    const size_t cap = std::min<size_t>({prompt.size(), history.size(),
                                         (size_t)std::max(cur_pos, 0)});
    size_t l = 0;
    while (l < cap && prompt[l] == history[l]) l++;
    d.L = (int)l;
    d.discard_above = d.L;
    const int pickable = std::min(d.L, (int)prompt.size() - 1);
    const int c = kvflash_ladder_pick(ladder_positions, pickable);
    if (c > 0) { d.C = c; d.replay_from = c; }
    return d;
}

// What a restore does with a snapshot. Dense: the existing path. Consume:
// pooled snapshot, pager blob restored, only the suffix prefills. AdoptDense:
// dense snapshot whose prompt outgrows the pool; its rows already sit in the
// pool tensor in the identity layout (chunk c in block c), so the pager adopts
// them by bookkeeping and the pooled prefill continues from the cut.
// FullReprefill: drop the snapshot and prefill the whole prompt from 0.
// Reject: fail the request (a pooled snapshot cannot be used and the full
// prompt is unknown).
enum class KvflashRestorePlan { Dense, Consume, FullReprefill, Reject, AdoptDense };

// A pooled snapshot resumes only at a whole pager chunk: a restored prefix
// that ends mid-chunk would need the partial chunk re-sealed. Anything that
// cannot consume must re-prefill the WHOLE prompt (never just the suffix).
// adopt_dense (live restore on) keeps a dense prefix across the pool boundary.
inline KvflashRestorePlan kvflash_restore_plan(bool consume_enabled, bool is_pooled,
                                               int kv_offset, int chunk_tokens,
                                               bool have_full_prompt,
                                               int total_tokens = 0, int pool_tokens = 0,
                                               bool adopt_dense = false) {
    if (!is_pooled) {
        // The pool tensor holds pool_tokens rows: a longer dense snapshot (a disk
        // entry from a server with no pool or a larger one) cannot be written
        // into it, not even to be adopted. Re-prefill whole.
        if (pool_tokens > 0 && kv_offset > pool_tokens)
            return have_full_prompt ? KvflashRestorePlan::FullReprefill
                                    : KvflashRestorePlan::Reject;
        // A dense snapshot cannot continue into a pooled prefill unless the
        // pager adopts its rows, so a prompt that outgrows the pool otherwise
        // re-prefills whole.
        if (have_full_prompt && pool_tokens > 0 && kv_offset > 0 &&
            total_tokens > kv_offset && total_tokens > pool_tokens - chunk_tokens)
            return adopt_dense ? KvflashRestorePlan::AdoptDense
                               : KvflashRestorePlan::FullReprefill;
        return KvflashRestorePlan::Dense;
    }
    if (consume_enabled && kv_offset > 0 && chunk_tokens > 0 &&
        kv_offset % chunk_tokens == 0) {
        return KvflashRestorePlan::Consume;
    }
    return have_full_prompt ? KvflashRestorePlan::FullReprefill
                            : KvflashRestorePlan::Reject;
}

// Prompt tokens the server's snapshot lets the request skip under `plan`.
inline int kvflash_server_cut(KvflashRestorePlan plan, int snap_pos) {
    return plan == KvflashRestorePlan::FullReprefill || plan == KvflashRestorePlan::Reject
        ? 0 : snap_pos;
}

// Server snapshot (restore_and_generate) or live session (S8a): the larger cut
// wins, ties keep the server snapshot (no state restore needed). A live cut of
// -1 means no usable rung.
enum class KvflashResume { Server, Live };
inline KvflashResume kvflash_pick_resume(int server_cut, int live_cut) {
    return live_cut > server_cut ? KvflashResume::Live : KvflashResume::Server;
}

// Prompt tokens a restore would skip: the larger of the server snapshot's cut
// (capped by the tokens the slot matches; 0 without a snapshot) and the live
// session's cut. Mirrors the choice restore_and_generate_impl makes, so a
// cache reports what the backend takes rather than what the snapshot holds.
inline int kvflash_restorable_tokens(bool has_snapshot, bool consume_enabled, bool is_pooled,
                                     int snap_pos, int slot_tokens, int chunk_tokens,
                                     int prompt_tokens, int pool_tokens,
                                     bool session_enabled, int live_cut) {
    const int server = has_snapshot
        ? std::min(slot_tokens,
                   kvflash_server_cut(kvflash_restore_plan(consume_enabled, is_pooled, snap_pos,
                                                           chunk_tokens, prompt_tokens > 0,
                                                           prompt_tokens, pool_tokens,
                                                           session_enabled),
                                      snap_pos))
        : 0;
    return std::max({server, live_cut, 0});
}

// Token history for [0, kv_offset + suffix): real ids for the restored prefix
// when the full prompt is known, zeros otherwise (the old behaviour).
inline std::vector<int32_t> kvflash_restore_history(
        const std::vector<int32_t> & suffix, int kv_offset,
        const std::vector<int32_t> * full_prompt) {
    std::vector<int32_t> h;
    if (kv_offset > 0) {
        if (full_prompt && (int)full_prompt->size() >= kv_offset) {
            h.assign(full_prompt->begin(), full_prompt->begin() + kv_offset);
        } else {
            h.assign((size_t)kv_offset, 0);
        }
    }
    h.insert(h.end(), suffix.begin(), suffix.end());
    return h;
}

// GGML M-RoPE consumes positions axis-major:
//   [all temporal][all height][all width][all extra].
// position_stride is the complete token width of the destination tensor;
// token_offset allows one request segment to be written into a packed batch.
inline void fill_qwen35_mrope_positions(int32_t * positions,
                                        int position_stride,
                                        int token_offset,
                                        int base_pos,
                                        int n_tokens) {
    for (int i = 0; i < n_tokens; ++i) {
        const int p = base_pos + i;
        const int row = token_offset + i;
        positions[0 * position_stride + row] = p;
        positions[1 * position_stride + row] = p;
        positions[2 * position_stride + row] = p;
        positions[3 * position_stride + row] = 0;
    }
}

inline void fill_qwen35_mrope_positions(int32_t * positions,
                                        int base_pos, int n_tokens) {
    fill_qwen35_mrope_positions(
        positions, n_tokens, /*token_offset=*/0, base_pos, n_tokens);
}

// Upload a causal mask into a mask tensor sized for max_ctx. Flash attention
// reads mask columns only inside the KV view, which is the window length
// rounded up to at most 256; one more 256 stride covers a kernel's final
// partial KV tile. Only those columns are built and copied (a strided 2-D
// write), so the cost follows the live context instead of max_ctx. Columns
// past them keep stale values that no kernel reads.
// LUCE_QWEN35_MASK_FULL_WIDTH=1 restores the full-width upload.
inline int qwen35_causal_mask_live_width(int kv_len, int full_width) {
    return std::min(full_width, align_up(kv_len, 256) + 256);
}

inline void upload_qwen35_causal_mask_window(ggml_tensor * mask, int kv_len,
                                             int n_tokens, int kv_start,
                                             int kq_stride_pad, int win_start) {
    if (!mask) return;
    static const bool full_width = [] {
        const char * e = std::getenv("LUCE_QWEN35_MASK_FULL_WIDTH");
        return e && e[0] == '1';
    }();
    const int full = (int)mask->ne[0];
    const int live = full_width ? full : qwen35_causal_mask_live_width(kv_len, full);
    std::vector<uint16_t> data;
    build_causal_mask(data, kv_len, n_tokens, kv_start, kq_stride_pad,
                      win_start, live);
    if (live == full) {
        ggml_backend_tensor_set(mask, data.data(), 0,
                                sizeof(uint16_t) * data.size());
        return;
    }
    const size_t row_bytes = sizeof(uint16_t) * (size_t)live;
    ggml_backend_tensor_set_2d(mask, data.data(), 0, row_bytes,
                               data.size() / (size_t)live, mask->nb[1], row_bytes);
}

inline void upload_qwen35_causal_mask(ggml_tensor * mask, int kv_start,
                                       int n_tokens, int kq_stride_pad) {
    upload_qwen35_causal_mask_window(mask, kv_start + n_tokens, n_tokens,
                                     kv_start, kq_stride_pad, /*win_start=*/0);
}

}  // namespace luce::common
