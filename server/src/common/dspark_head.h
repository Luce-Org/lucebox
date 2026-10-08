#pragma once

#include "dflash_target.h"
#include "internal.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace luce::common {

// Drafter lifecycle generation. Every drafter free/reload bumps it; the fused
// chain-graph cache keys on it so a reloaded drafter that lands at the same
// addresses can never reuse a graph built over freed weights. The counter is
// process-wide, atomic, and monotonic.
void dspark_note_drafter_lifecycle();
uint64_t dspark_drafter_generation();
// Number of fused chain graphs built so far (diagnostic; lets tests assert a
// cache hit or miss without reaching into the cache).
uint64_t dspark_chain_graph_build_count();

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
// separate state retain the legacy behavior by leaving it null.
bool dspark_markov_correct_greedy_chain_fused(const DraftWeights & dw,
                                              ggml_backend_t backend,
                                              ggml_tensor * lm_head,
                                              const float * local_hidden,
                                              int q_len,
                                              int32_t last_tok,
                                              std::vector<int32_t> & draft_tok,
                                              std::vector<float> * confidence_out = nullptr,
                                              const float * confidence_hidden = nullptr,
                                              std::vector<float> * logit_margin_out = nullptr);

// The fused chain with the vocabulary split across ranks that run the same
// graph in lockstep (a cluster's head and workers). Each rank projects rows
// [row_offset, row_offset + n_rows) of lm_head and of the Markov bias, so the
// lm_head read is divided between the ranks' devices. The model supplies the
// collectives as graph adapters:
//   share(ctx, hidden) -> the root's hidden columns on every rank, exactly;
//   pick(ctx, logits)  -> I32 [1], the argmax over every rank's slice
//                         (ties to the lowest index, as ggml_argmax).
// The tokens equal the unsplit chain's: each vocabulary row is the same product
// in either form. Every rank must call this the same number of times with the
// same n_candidates, in the same order relative to its other collectives.
// local_hidden (padded as in the fused chain) is read on the root only and may
// be null elsewhere. read = false enqueues the graph and returns at once: a rank
// that only takes part, whose stream orders the graph before its next work.
struct DsparkVocabSplit {
    int64_t row_offset = 0;
    int64_t n_rows = 0;
    ggml_tensor * markov_w1 = nullptr;   // [markov_rank, vocab], on `backend`'s device
    ggml_tensor * markov_w2 = nullptr;   // [markov_rank, >= n_rows]: its first n_rows rows are this rank's
    std::function<ggml_tensor *(ggml_context *, ggml_tensor *)> share;
    std::function<ggml_tensor *(ggml_context *, ggml_tensor *)> pick;
    const void * id = nullptr;           // identifies share/pick for the graph cache
};

bool dspark_markov_chain_vocab_split(const DraftWeights & dw,
                                     ggml_backend_t backend,
                                     ggml_tensor * lm_head,
                                     const DsparkVocabSplit & split,
                                     const float * local_hidden,
                                     int n_candidates,
                                     int32_t last_tok,
                                     bool read,
                                     std::vector<int32_t> & draft_tok,
                                     std::vector<float> * confidence_out = nullptr,
                                     const float * confidence_hidden = nullptr);

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

}  // namespace luce::common
