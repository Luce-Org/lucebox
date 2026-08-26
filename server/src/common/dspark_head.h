#pragma once

#include "dflash_target.h"
#include "internal.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// Append an anchor-first greedy DSpark Markov chain to an existing graph.
// `base_logits` is [vocab, positions], `seed` is one I32 token, and each
// returned scalar token remains device-resident to condition the next row.
// This lets persistent drafter graphs include LM head + Markov correction in
// their single backend compute instead of launching a second graph.
bool dspark_build_greedy_markov_chain(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const DraftWeights & dw,
    ggml_tensor * base_logits,
    ggml_tensor * seed,
    std::vector<ggml_tensor *> & tokens_out);

// Reusable graph/allocation for the device-resident Markov head. It is owned
// by one model backend and must be reset before either model's weights or GPU
// backend are released.
class DsparkMarkovChainCache {
public:
    DsparkMarkovChainCache();
    ~DsparkMarkovChainCache();

    DsparkMarkovChainCache(const DsparkMarkovChainCache &) = delete;
    DsparkMarkovChainCache & operator=(const DsparkMarkovChainCache &) = delete;

    void reset();

private:
    struct Impl;
    Impl * impl_ = nullptr;

    friend bool dspark_markov_correct_greedy_chain_fused_tensor(
        const DraftWeights &, ggml_backend_t, ggml_tensor *, ggml_tensor *,
        int, int32_t, std::vector<int32_t> &, DsparkMarkovChainCache &);
};

bool dspark_markov_correct_greedy_chain(const DraftWeights & dw,
                                        ggml_backend_t backend,
                                        DFlashTarget & target,
                                        const float * local_hidden,
                                        int q_len,
                                        int32_t last_tok,
                                        float confidence_threshold,
                                        std::vector<int32_t> & draft_tok);

// Fused variant: base logits (one lm_head matmul over all candidates) +
// unrolled Markov correction chain + in-graph argmax feeding the next
// step's get_rows, all in ONE graph on the draft backend. No host logits
// round-trip. When confidence_out is non-null and the checkpoint has a
// compatible confidence head, returns one score per candidate from the same
// graph and host synchronization as the token ids. `confidence_hidden`, when
// non-null, has the same padded layout as `local_hidden` and supplies the
// pre-output-norm state expected by the confidence head. Callers without a
// separate state retain the legacy behavior by leaving it null. The returned
// vector is always root-first: one exact `last_tok`, followed by either
// q_len-1 legacy DFlash candidates or q_len anchor-first DSpark candidates.
bool dspark_markov_correct_greedy_chain_fused(const DraftWeights & dw,
                                              ggml_backend_t backend,
                                              ggml_tensor * lm_head,
                                              const float * local_hidden,
                                              int q_len,
                                              int32_t last_tok,
                                              std::vector<int32_t> & draft_tok,
                                              std::vector<float> * confidence_out = nullptr,
                                              const float * confidence_hidden = nullptr);

// Same fused chain, but consumes an already-resident F32 drafter tensor.
// This is the single-GPU fast path: the LM head reads the drafter output
// directly, avoiding the blocking GPU -> host -> GPU hidden-state round trip.
// `local_hidden` must have shape [dw.n_embd, >= q_len] on `backend`.
bool dspark_markov_correct_greedy_chain_fused_tensor(
    const DraftWeights & dw,
    ggml_backend_t backend,
    ggml_tensor * lm_head,
    ggml_tensor * local_hidden,
    int q_len,
    int32_t last_tok,
    std::vector<int32_t> & draft_tok,
    DsparkMarkovChainCache & cache);

// DDTree candidate generation with the Markov correction: base logits for
// all n_tokens positions in ONE lm_head matmul; rows 1..n-1 get the low-rank
// previous-token bias chained along the main (argmax) path; top-K extracted
// on host via extract_draft_topk. Output contract matches
// DFlashTarget::project_hidden_to_topk (row 0 = seed position, uncorrected).
bool dspark_markov_project_topk(const DraftWeights & dw,
                                ggml_backend_t backend,
                                ggml_tensor * lm_head,
                                const float * hidden,
                                int n_tokens, int K, float temperature,
                                int32_t last_tok,
                                std::vector<float> & top_log_probs,
                                std::vector<int32_t> & top_token_ids);

}  // namespace dflash::common
