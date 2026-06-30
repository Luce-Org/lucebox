#pragma once

#include <cstdint>

namespace dflash::common {

bool deepseek4_cuda_hc_pre_mix(const float * hc_state_host,
                               const void *  fn_device,
                               int           n_embd,
                               int           n_hc,
                               float         eps,
                               float *       mix_host);

bool deepseek4_cuda_hc_pre(const float * hc_state_host,
                           const void *  fn_device,
                           const float * scale_host,
                           const float * base_host,
                           int           n_embd,
                           int           n_hc,
                           int           sinkhorn_iters,
                           float         eps,
                           float *       working_host,
                           float *       post_host,
                           float *       comb_host);

bool deepseek4_cuda_hc_pre_device(const void * hc_state_device,
                                  const void * fn_device,
                                  const void * scale_device,
                                  const void * base_device,
                                  int          n_embd,
                                  int          n_hc,
                                  int          sinkhorn_iters,
                                  float        eps,
                                  void *       working_device,
                                  void *       post_device,
                                  void *       comb_device);

bool deepseek4_cuda_hc_pre_device_params(const void * hc_state_device,
                                         const void * fn_device,
                                         const float * scale_host,
                                         const float * base_host,
                                         int           n_embd,
                                         int           n_hc,
                                         int           sinkhorn_iters,
                                         float         eps,
                                         void *        working_device,
                                         void *        post_device,
                                         void *        comb_device);

} // namespace dflash::common
