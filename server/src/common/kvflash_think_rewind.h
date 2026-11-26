#pragma once

// KVFlash Mode 5 + Think-then-Rewind-to-R: pure planning logic (no ggml, no GPU).
//
// After Mode 5 (PFlash v3) prefills [R..prompt_end) and AR decode generates
// N think tokens (or hits early EOS), the backend scores all sealed chunks
// [0..R) via fused GPU exact attention mass over prompt_tail + <think> queries.
//
// This header constructs the Mode 5 U exact-mass union (half the recall budget
// from Mode 5, remainder from exact-mass top-K) and applies confidence gating
// for normal triggers and early-EOS hallucination catches.

#include "kvflash_prefill_reselect.h"

#include <algorithm>
#include <functional>
#include <vector>

namespace luce::common {

struct KvFlashThinkRewindCfg {
    int top_k_check               = 8;  // top-K exact-mass units checked for evicted-mass gate
    int min_top_evicted           = 0;  // normal trigger: >= this many top-K units evicted
    int early_eos_min_top_evicted = 2;  // early-EOS trigger: stronger bar to catch short hallucinations
    int min_new_evicted           = 1;  // union must page in at least this many evicted units
};

struct KvFlashThinkRewindPlan {
    bool               should_rewind = false;
    int                keep_m5 = 0;
    int                top_evicted = 0;
    int                new_evicted = 0;
    std::vector<int>   m4_top;        // ranked exact-mass units (size <= total_u)
    std::vector<int>   union_units;   // final union unit IDs (size <= total_u)
    std::vector<float> union_scores;  // chunk-level scores encoding union_units verbatim
};

inline KvFlashThinkRewindPlan kvflash_think_rewind_plan(
        const std::vector<int> & m5_picked,
        const std::vector<float> & m4_chunk_scores,
        const KvFlashRecallIn & rin,
        int sealed_chunks,
        bool is_early_eos,
        const KvFlashThinkRewindCfg & cfg,
        const std::function<bool(int unit_id)> & is_unit_evicted) {
    KvFlashThinkRewindPlan out;
    const int unit = rin.unit;
    const int total_u = unit > 0 ? rin.recall_chunks / unit : 0;
    if (total_u <= 0 || sealed_chunks <= 0) return out;

    const int first_chunk  = kvflash_recall_first_chunk(rin.pinned_chunks);
    const int first_window = kvflash_recall_first_window(rin);
    out.m4_top = kvflash_recall_pick(m4_chunk_scores, first_window, total_u, unit, first_chunk);
    if (out.m4_top.empty()) return out;

    const int budget_u = std::min(total_u, (int)out.m4_top.size());
    out.keep_m5 = std::min(budget_u / 2, (int)m5_picked.size());

    const int top_k = std::min(std::max(1, cfg.top_k_check), (int)out.m4_top.size());
    for (int i = 0; i < top_k; ++i) {
        if (is_unit_evicted(out.m4_top[(size_t)i])) out.top_evicted++;
    }

    auto has_u = [&](int b) {
        return std::find(out.union_units.begin(), out.union_units.end(), b) != out.union_units.end();
    };
    out.union_units.reserve((size_t)budget_u);
    for (int i = 0; i < out.keep_m5; ++i) {
        const int b = m5_picked[(size_t)i];
        if (!has_u(b)) out.union_units.push_back(b);
    }
    for (int b : out.m4_top) {
        if ((int)out.union_units.size() >= budget_u) break;
        if (!has_u(b)) out.union_units.push_back(b);
    }
    for (int b : m5_picked) {
        if ((int)out.union_units.size() >= budget_u) break;
        if (!has_u(b)) out.union_units.push_back(b);
    }
    for (int b : out.union_units) {
        if (is_unit_evicted(b)) out.new_evicted++;
    }

    const int req_top = is_early_eos ? cfg.early_eos_min_top_evicted : cfg.min_top_evicted;
    out.should_rewind = (out.top_evicted >= req_top) && (out.new_evicted >= cfg.min_new_evicted);

    if (out.should_rewind) {
        out.union_scores.assign((size_t)sealed_chunks, 0.0f);
        for (size_t idx = 0; idx < out.union_units.size(); ++idx) {
            const int b = out.union_units[idx];
            const float sv = 1000.0f - (float)idx;
            for (int c = b * unit; c < (b + 1) * unit && c < sealed_chunks; ++c) {
                if (c >= 0) out.union_scores[(size_t)c] = sv;
            }
        }
    }
    return out;
}

}  // namespace luce::common
