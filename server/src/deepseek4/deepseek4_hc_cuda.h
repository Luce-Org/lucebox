#pragma once

#include <cstddef>
#include <cstdint>

namespace dflash::common {

// CUDA and HIP select devices per host thread. Call this at worker/shard entry
// before invoking the direct HC helpers.
bool deepseek4_cuda_hc_set_device(int device);

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

// Persistent F16 device mirrors for quantized HC fn weights: upload the host
// F16 copy to the given device, and free it on that device later.
bool deepseek4_cuda_hc_upload_f16(int          device,
                                  const void * host_f16,
                                  size_t       bytes,
                                  void **      device_out);

void deepseek4_cuda_hc_free(int device, void * device_ptr);

} // namespace dflash::common
