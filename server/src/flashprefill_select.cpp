// Block selection for FlashPrefill: turns per-(q_block, k_block, head) scores
// into a sparse, sorted index list per (q_block, head). This runs on host
// because the score grid is small (~1100×1100×16 = 19 M entries at 140K ctx)
// and the logic is sequential. The selected indices feed the sparse
// flash_forward CUDA kernel (kernel 4).
//
// Selection rules (from qhfan/FlashPrefill):
//   - sink:       k_block_idx < attention_sink (always include)
//   - window:     q_block_idx - window < k_block_idx <= q_block_idx
//   - last_full:  q_block_idx >= M - last_n_full → include all
//   - top-K dyn:  score >= max_score * alpha
//   - causal:     k_block_idx <= q_block_idx
//
// Output: compact_indices[B, M, N, H] (padded with -1) and counts[B, M, H].

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dflash::common {
namespace flashprefill {

// score: [B, M, N, H] row-major (B outer, H fastest).
// idx_out: [B, M, N, H] same layout, padded with -1.
// cnt_out: [B, M, H] same layout.
void block_select_host(
    const float * score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int32_t * idx_out, int32_t * cnt_out)
{
    // Score strides assume contiguous [B, M, N, H] row-major.
    const int s_b = M * N * H;
    const int s_m = N * H;
    const int s_n = H;
    const int s_h = 1;
    const int idx_s_b = M * N * H;
    const int idx_s_m = N * H;
    const int idx_s_n = H;
    const int idx_s_h = 1;
    const int cnt_s_b = M * H;
    const int cnt_s_m = H;
    const int cnt_s_h = 1;

    std::vector<int32_t> selected;
    selected.reserve(N);

    for (int b = 0; b < B; ++b) {
        for (int m = 0; m < M; ++m) {
            for (int h = 0; h < H; ++h) {
                selected.clear();

                // Find max score for this (b, m, h) across n in [0, m].
                float max_score = -INFINITY;
                for (int n = 0; n <= m; ++n) {
                    float v = score[b*s_b + m*s_m + n*s_n + h*s_h];
                    if (v > max_score) max_score = v;
                }
                const float thresh = max_score * alpha;
                const bool last_full = (m >= M - last_n_full);

                for (int n = 0; n <= m; ++n) {
                    bool keep = false;
                    if (n < attention_sink) keep = true;
                    if (m - n < window && n <= m) keep = true;
                    if (last_full) keep = true;
                    if (!keep) {
                        float v = score[b*s_b + m*s_m + n*s_n + h*s_h];
                        if (v >= thresh) keep = true;
                    }
                    if (keep) selected.push_back((int32_t)n);
                }
                std::sort(selected.begin(), selected.end());

                int32_t * idx_row = idx_out + b*idx_s_b + m*idx_s_m + h*idx_s_h;
                for (int n = 0; n < N; ++n) {
                    idx_row[n*idx_s_n] = (n < (int)selected.size()) ? selected[n] : -1;
                }
                cnt_out[b*cnt_s_b + m*cnt_s_m + h*cnt_s_h] = (int32_t)selected.size();
            }
        }
    }
}

} // namespace flashprefill
} // namespace dflash::common
