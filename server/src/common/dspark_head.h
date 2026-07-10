#pragma once

#include "dflash_target.h"
#include "internal.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

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
// graph and host synchronization as the token ids.
bool dspark_markov_correct_greedy_chain_fused(const DraftWeights & dw,
                                              ggml_backend_t backend,
                                              ggml_tensor * lm_head,
                                              const float * local_hidden,
                                              int q_len,
                                              int32_t last_tok,
                                              std::vector<int32_t> & draft_tok,
                                              std::vector<float> * confidence_out = nullptr);

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
