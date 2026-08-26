#include "kda_replayssm_cuda.h"

#include <cuda_runtime.h>

namespace dflash::common {
namespace {

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
constexpr int kWarp = 64;
#else
constexpr int kWarp = 32;
#endif

constexpr int kStateDim = 128;
constexpr int kColumnsPerBlock = 4;

__device__ __forceinline__ float replay_warp_sum(float value) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffU, value, offset, kWarp);
    }
    return value;
}

// Each logical warp owns one state column. The row dimension is distributed
// over its lanes exactly like ggml-cuda's KDA recurrence, preserving the same
// reduction and update order while avoiding a dense state checkpoint.
__global__ void __launch_bounds__(kWarp * kColumnsPerBlock, 2)
kda_replayssm_commit_kernel(float * const * state_ptrs,
                            const float * __restrict__ k,
                            const float * __restrict__ v,
                            const float * __restrict__ g,
                            const float * __restrict__ beta,
                            int accepted,
                            int n_heads,
                            int n_layers) {
    const int head = blockIdx.x;
    const int layer = blockIdx.y;
    const int column = blockIdx.z * kColumnsPerBlock + threadIdx.y;
    const int lane = threadIdx.x;
    if (column >= kStateDim) return;

    constexpr int kRowsPerLane = kStateDim / kWarp;
    float * state = state_ptrs[layer] +
        ((size_t)head * kStateDim + column) * kStateDim;
    float state_shard[kRowsPerLane];
#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
        state_shard[row] = state[row * kWarp + lane];
    }

    for (int token = 0; token < accepted; ++token) {
        const size_t head_token =
            (size_t)head + (size_t)n_heads *
            ((size_t)layer + (size_t)n_layers * token);
        const float * k_token = k + head_token * kStateDim;
        const float * v_token = v + head_token * kStateDim;
        const float * g_token = g + head_token * kStateDim;
        const float beta_value = beta[head_token];

        float k_reg[kRowsPerLane];
        float kv_shard = 0.0f;
#pragma unroll
        for (int row = 0; row < kRowsPerLane; ++row) {
            const int i = row * kWarp + lane;
            k_reg[row] = k_token[i];
            kv_shard += expf(g_token[i]) * state_shard[row] * k_reg[row];
        }
        const float kv_column = replay_warp_sum(kv_shard);
        const float delta = (v_token[column] - kv_column) * beta_value;

#pragma unroll
        for (int row = 0; row < kRowsPerLane; ++row) {
            const int i = row * kWarp + lane;
            state_shard[row] =
                expf(g_token[i]) * state_shard[row] + k_reg[row] * delta;
        }
    }

#pragma unroll
    for (int row = 0; row < kRowsPerLane; ++row) {
        state[row * kWarp + lane] = state_shard[row];
    }
}

}  // namespace

bool kda_replayssm_commit_async(float * const * state_ptrs_dev,
                                const float * k,
                                const float * v,
                                const float * g,
                                const float * beta,
                                int accepted,
                                int state_dim,
                                int n_heads,
                                int n_layers,
                                int max_tokens,
                                void * stream,
                                bool * launched) {
    if (launched) *launched = false;
    if (!state_ptrs_dev || !k || !v || !g || !beta ||
        accepted <= 0 || accepted > max_tokens ||
        state_dim != kStateDim || n_heads <= 0 || n_layers <= 0) {
        return false;
    }
    if (n_heads > 65535 || n_layers > 65535) return false;

    const dim3 block(kWarp, kColumnsPerBlock);
    const dim3 grid((unsigned)n_heads, (unsigned)n_layers,
                    (kStateDim + kColumnsPerBlock - 1) / kColumnsPerBlock);
    (void)cudaGetLastError();
    kda_replayssm_commit_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        state_ptrs_dev, k, v, g, beta, accepted, n_heads, n_layers);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (launched) *launched = true;
    return true;
}

}  // namespace dflash::common
