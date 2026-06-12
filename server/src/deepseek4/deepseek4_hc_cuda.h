#pragma once

#include <cstdint>

namespace dflash::common {

bool deepseek4_cuda_hc_pre_mix(const float * hc_state_host,
                               const void *  fn_device,
                               int           n_embd,
                               int           n_hc,
                               float         eps,
                               float *       mix_host);

} // namespace dflash::common
