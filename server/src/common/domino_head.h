#pragma once

#include "dflash_target.h"
#include "internal.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// Fused variant: one GPU graph = lm_head projection of the candidate hidden
// states + unrolled GRU correction chain with in-graph argmax -> get_rows
// token feedback. Requires the target to expose its lm_head and a GPU (f16)
// token-embedding table. Runs on a dedicated CUDA backend instance so the
// ggml-cuda graph cache can replay it across steps.
// [TAG_FUSED_LOOP] When hidden_dev is set (the draft graph's device-resident
// hidden_states on the same backend), the head reads rows 1..q_len-1 in place
// and local_hidden may be null: no D2H hidden readback, no H2D re-upload.
bool domino_correct_greedy_chain_fused(const DraftWeights & dw,
                                       ggml_backend_t backend,
                                       ggml_tensor * lm_head,
                                       ggml_tensor * embd_table,
                                       const float * local_hidden,
                                       int q_len,
                                       int32_t last_tok,
                                       std::vector<int32_t> & draft_tok,
                                       int cand_k = 0,
                                       std::vector<float> * cand_probs = nullptr,
                                       std::vector<int32_t> * cand_ids = nullptr,
                                       ggml_tensor * hidden_dev = nullptr);

bool domino_correct_greedy_chain(const DraftWeights & dw,
                                 ggml_backend_t backend,
                                 DFlashTarget & target,
                                 const float * local_hidden,
                                 int q_len,
                                 int32_t last_tok,
                                 std::vector<int32_t> & draft_tok);

}  // namespace dflash::common
