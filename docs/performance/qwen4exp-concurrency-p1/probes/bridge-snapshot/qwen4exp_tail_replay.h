#pragma once

#include "qwen4exp/qwen4exp_cache.h"

#include <cstdint>
#include <string>

namespace luce::common {

// Harness-only reconstruction of the recurrent state at B=992 from the
// native boundary dumps emitted by QWEN4EXP_DUMP_BIN.
bool qwen4exp_tail_replay_992_100(ggml_backend_t backend,
                                  const Qwen4ExpWeights & weights,
                                  Qwen4ExpCache & cache,
                                  const int32_t * tokens,
                                  std::string & error);

} // namespace luce::common
