// Shared prefill helpers for Qwen3.5/3.6.

#pragma once

#include "attn_masks.h"
#include "ggml-backend.h"

#include <algorithm>
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
inline int kvflash_prefill_batch_end(int pos, int end, int grid, int max_batch) {
    const int g = max_batch < grid ? std::max(1, max_batch) : grid;
    const int maxb = (std::max(max_batch, 1) / g) * g;
    return std::min(end, ((pos + maxb) / g) * g);
}

inline std::vector<std::pair<int, int>> kvflash_prefill_batches(int start, int end, int grid,
                                                                int max_batch) {
    std::vector<std::pair<int, int>> out;
    for (int pos = start; pos < end;) {
        const int e = kvflash_prefill_batch_end(pos, end, grid, max_batch);
        out.emplace_back(pos, e);
        pos = e;
    }
    return out;
}

// LUCE_KVFLASH_RESTORE_CONSUME=1 opts in to carrying a pooled KVFlash session
// across turns: pooled prefix snapshots are saved and a restore consumes them
// instead of re-prefilling the prompt. Default OFF.
inline bool kvflash_restore_consume_enabled() {
    const char * v = std::getenv("LUCE_KVFLASH_RESTORE_CONSUME");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
}

// What a restore does with a snapshot. Dense: the existing path. Consume:
// pooled snapshot, pager blob restored, only the suffix prefills. FullReprefill:
// drop the snapshot and prefill the whole prompt from 0. Reject: fail the
// request (a pooled snapshot cannot be used and the full prompt is unknown).
enum class KvflashRestorePlan { Dense, Consume, FullReprefill, Reject };

// A pooled snapshot resumes only at a whole pager chunk: a restored prefix
// that ends mid-chunk would need the partial chunk re-sealed. Anything that
// cannot consume must re-prefill the WHOLE prompt (never just the suffix).
inline KvflashRestorePlan kvflash_restore_plan(bool consume_enabled, bool is_pooled,
                                               int kv_offset, int chunk_tokens,
                                               bool have_full_prompt) {
    if (!is_pooled) return KvflashRestorePlan::Dense;
    if (consume_enabled && kv_offset > 0 && chunk_tokens > 0 &&
        kv_offset % chunk_tokens == 0) {
        return KvflashRestorePlan::Consume;
    }
    return have_full_prompt ? KvflashRestorePlan::FullReprefill
                            : KvflashRestorePlan::Reject;
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
