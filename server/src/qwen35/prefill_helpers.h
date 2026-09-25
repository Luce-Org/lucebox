// Shared prefill helpers for Qwen3.5/3.6.

#pragma once

#include "attn_masks.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace luce::common {

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
