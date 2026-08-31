#pragma once

#include "dflash_target.h"
#include "internal.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// Invalidate the cached selector projection graph. MUST be called whenever
// draft weights are freed (park); the cache is keyed on the DraftWeights
// address, which is stable across a park/unpark reload of the same backend
// member, so without this the cached graph would keep pointers to freed
// selector weight tensors.
void dflash2_selector_graph_invalidate();
uint64_t dflash2_selector_graph_generation();

// DFlash 2 candidate selector for greedy chain drafting.
//
// For every drafted block position the target lm_head logits are reduced to
// the selector's top-k candidates (log-probs, so per-position constants do
// not matter for the argmax), then one path is traced through them:
//   score(c) = logp(c) + < pred_cb[prev] * hproj(h_pos), succ_cb[c] >
//   prev     = argmax_c score(c)
// starting from the block seed `last_tok`. Runs the projections (hproj GEMV
// and codebook row gathers) in one small graph on `backend`, the k-way path
// search on the host. Fills draft_tok = [last_tok, tok_1 .. tok_{q_len-1}].
bool dflash2_select_chain(const DraftWeights & dw,
                          ggml_backend_t backend,
                          DFlashTarget & target,
                          const float * local_hidden,
                          int q_len,
                          int32_t last_tok,
                          std::vector<int32_t> & draft_tok);

// Same selector, batched over host-resident drafter hidden blocks and using a
// local target lm_head tensor. The expensive lm_head projection covers every
// (lane, depth) in one graph, GPU top-K is invoked once, and selector
// projections/readback are shared across the cohort.
bool dflash2_select_chains_batched(
    const DraftWeights & dw,
    ggml_backend_t backend,
    ggml_tensor * lm_head,
    const std::vector<const float *> & hidden_by_lane,
    int q_len,
    const std::vector<int32_t> & last_tokens,
    std::vector<std::vector<int32_t>> & draft_tokens);

// Selector-scored candidates for DDTree construction (DARTree-style): the
// same per-position top-k + selector projections as the chain path, kept on
// the host so the tree builder can ask for branch-conditioned scores.
struct Dflash2TreeScores {
    int n_cand = 0, K = 0, rank = 0;
    int32_t seed = 0;
    std::vector<float>   lp;     // [n_cand*K]
    std::vector<int32_t> ids;    // [n_cand*K]
    std::vector<float>   hproj;  // [rank*n_cand]
    std::vector<float>   succ;   // [rank*n_cand*K]
    std::vector<float>   pred;   // [rank*(1+n_cand*K)]

    // K selector-adjusted scores for position `depth-1`, conditioned on the
    // prefix'/s last token. Sorted descending; false if depth out of range.
    bool topk(const std::vector<int32_t> & prefix, int next_depth,
              std::vector<float> & out_lp, std::vector<int32_t> & out_ids) const;
};

bool dflash2_score_candidates(const DraftWeights & dw,
                              ggml_backend_t backend,
                              DFlashTarget & target,
                              const float * local_hidden,
                              int q_len,
                              int32_t last_tok,
                              float temperature,
                              Dflash2TreeScores & out);

}  // namespace dflash::common
