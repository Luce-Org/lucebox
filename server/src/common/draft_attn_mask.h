#pragma once

// Pure construction of the draft-KV SWA attention mask (f16 bit patterns,
// [kv_total, q_len] row-major, row = query). Columns: [0, cap) are the ctx
// ring slots (slot_pos[s] = absolute position held, -1 = empty), [cap, cap+q_len)
// are the block's noise rows, the rest is alignment padding (always masked).
//
//   causal == true  (DFlash 1 / Qwen3.6 drafters; unchanged legacy behaviour):
//     ctx window anchored at `committed` and shared by every query row,
//     noise columns causal (j <= q).
//   causal == false (DFlash 2 upstream GGUF, attention.causal = false):
//     noise columns fully visible; the ctx window is per query row,
//     visible iff q_pos - k_pos < window with q_pos = committed + q.

#include <algorithm>
#include <cstdint>
#include <vector>

namespace luce::common {

constexpr uint16_t kDraftMaskF16Zero   = 0x0000;
constexpr uint16_t kDraftMaskF16NegInf = 0xFC00;

inline std::vector<uint16_t> build_draft_swa_mask(
        int cap, int kv_total, int q_len, int committed,
        const int32_t * slot_pos, int window, bool causal) {
    std::vector<uint16_t> m((size_t)kv_total * q_len, kDraftMaskF16NegInf);
    const int win = std::min(committed, cap);
    const int lo  = committed - win;               // oldest position the ring can hold
    for (int q = 0; q < q_len; q++) {
        uint16_t * row = m.data() + (size_t)q * kv_total;
        int row_lo;
        if (causal) {
            const int eff_win = (window > 0 && win > window) ? window : win;
            row_lo = committed - eff_win;
        } else {
            row_lo = window > 0 ? std::max(lo, committed + q - window + 1) : lo;
        }
        for (int s = 0; s < cap; s++) {
            const int32_t p = slot_pos[s];
            if (p >= row_lo && p < committed) row[s] = kDraftMaskF16Zero;
        }
        for (int j = 0; j < q_len; j++)
            if (!causal || j <= q) row[cap + j] = kDraftMaskF16Zero;
    }
    return m;
}

} // namespace luce::common
