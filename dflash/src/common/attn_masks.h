// Attention mask builders for flash-attention kernels.
//
// Builds f16 masks aligned to the KQ stride pad required by ggml_flash_attn_ext.
// Used for both causal (prefill/decode) and tree-structured (DDTree verify) masks.

#pragma once

#include "ddtree.h"

#include <cstdint>
#include <vector>

namespace dflash27b {

// Minimum alignment required by ggml flash_attn_ext for mask rows.
static constexpr int KQ_MASK_PAD = 32;

// F16 encoding constants.
static constexpr uint16_t F16_ZERO    = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

inline int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

// Build a standard causal mask (lower-triangular) for n_tokens queries
// attending to kv_len keys starting at kv_start.
// kq_stride_pad: alignment for kv dimension (32 for F16/Q8, 256 for TurboQuant).
// win_start: optional window start for sliding-window attention.
inline void build_causal_mask(std::vector<uint16_t> & out,
                              int kv_len, int n_tokens, int kv_start,
                              int kq_stride_pad,
                              int win_start = 0) {
    const int kv_pad = align_up(kv_len, kq_stride_pad);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    const int abs_end = win_start + kv_len;
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        const int min_k = std::max(0, win_start);
        const int max_k = abs_q;
        for (int k = min_k; k <= max_k && k < abs_end; k++) {
            out[(size_t)q * kv_pad + (k - win_start)] = F16_ZERO;
        }
    }
}

// Build an ancestor-only attention mask for DDTree tree-structured verify.
// Each query position i can attend to its ancestors in the tree (including
// itself) plus all past KV positions.
//
// kv_pad_override: when nonzero, pin the kv (column) stride to this value
// instead of the helper's natural `align_up(past_length + N - win_start,
// kq_stride_pad)`.  Needed when the consumer tensor was allocated with a
// fixed kv extent (e.g. build_target_step_tree sizes sg.attn_mask at
// align_up(cache.max_ctx + N, kq_stride_pad)) and the helper-computed
// stride would not match the tensor's actual row pitch.  Default 0 keeps
// existing behavior.
inline void build_tree_mask(const DDTree & tree, int past_length,
                            std::vector<uint16_t> & out_mask,
                            int kq_stride_pad,
                            int win_start = 0,
                            int kv_pad_override = 0) {
    const int N       = 1 + tree.n_nodes;
    const int win_len = past_length + N - win_start;
    const int kv_pad  = kv_pad_override > 0
                          ? align_up(kv_pad_override, kq_stride_pad)
                          : align_up(win_len, kq_stride_pad);
    const int q_pad   = align_up(N, KQ_MASK_PAD);
    out_mask.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < N; q++) {
        for (int k = std::max(0, win_start); k < past_length; k++) {
            out_mask[(size_t)q * kv_pad + (k - win_start)] = F16_ZERO;
        }
        for (int j = 0; j < N; j++) {
            if (tree.visibility[(size_t)q * N + j]) {
                int col = past_length + j - win_start;
                if (col >= 0 && col < kv_pad) {
                    out_mask[(size_t)q * kv_pad + col] = F16_ZERO;
                }
            }
        }
    }
}

}  // namespace dflash27b
