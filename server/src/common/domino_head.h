#pragma once

#include "dflash_target.h"
#include "internal.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

bool domino_correct_greedy_chain(const DraftWeights & dw,
                                 ggml_backend_t backend,
                                 DFlashTarget & target,
                                 const float * local_hidden,
                                 int q_len,
                                 int32_t last_tok,
                                 std::vector<int32_t> & draft_tok);

}  // namespace dflash::common
