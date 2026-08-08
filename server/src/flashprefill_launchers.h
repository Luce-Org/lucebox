#pragma once

// Internal ABI shared by the FlashPrefill orchestration layer and the
// architecture-specific CUDA/HIP launchers.  Keep these declarations in one
// header: a return-type mismatch across separate `extern "C"` declarations is
// not diagnosed by the linker and previously made successful CUDA launches
// look like failures on aarch64/GB10.

#include "device_runtime.h"

namespace dflash::common::flashprefill {

extern "C" int launch_compute_mean_vector_bf16(
    const void * K, void * mean_K,
    int batch, int seq_len, int n_kv_heads, int head_dim, int block_size,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    cudaStream_t stream);

extern "C" int launch_compute_block_score_bf16(
    const void * Q, const void * mean_K, float sm_scale,
    void * score, void * score_max,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h,
    cudaStream_t stream);

} // namespace dflash::common::flashprefill
