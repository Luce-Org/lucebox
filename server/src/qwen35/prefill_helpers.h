// Shared prefill helpers for Qwen3.5/3.6.

#pragma once

#include "attn_masks.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace dflash::common {

inline int qwen35_prefill_ubatch(int fallback) {
    const char * value = std::getenv("DFLASH27B_PREFILL_UBATCH");
    return value ? std::max(1, std::atoi(value)) : fallback;
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

inline void upload_qwen35_causal_mask(ggml_tensor * mask, int kv_start,
                                       int n_tokens, int kq_stride_pad) {
    if (!mask) return;
    std::vector<uint16_t> data;
    build_causal_mask(data, kv_start + n_tokens, n_tokens, kv_start,
                      kq_stride_pad, /*win_start=*/0, (int)mask->ne[0]);
    ggml_backend_tensor_set(mask, data.data(), 0,
                            sizeof(uint16_t) * data.size());
}

}  // namespace dflash::common
