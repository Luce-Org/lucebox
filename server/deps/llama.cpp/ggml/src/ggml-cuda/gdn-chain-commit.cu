// SINGLE_REPLAY: commit of a single-sequence chain
// verify's replay log (ggml-cuda-gdn-chain.h). A lean path next to the paged
// commit in gdn-replay-log.cu: no pointer-attribute validation, no side stream
// and no synchronize, one launch for every layer, up to 16 tokens.
//
// Bitwise contract. The grouped-cols GDN kernel (gated_delta_net.cu,
// gated_delta_net_cuda_grouped_cols) updates each state element of head h,
// for token t, as
//     state_shard[c][r] = fmaf(k_reg[r], delta[c], g_val * state_shard[c][r])
// and records into its replay log, from the registers that update reads:
//     log[t][h][0]           = g_val      (the value every warp shuffles)
//     log[t][h][1 + row]     = k_val      (k_reg[r] = k_val = k_t[row])
//     log[t][h][1 + S + col] = delta_val  (lane 0's value, broadcast as delta[c])
// The kernel below applies the same expression to the same operands in the
// same token order, starting from the same element of the base state, so
// after n tokens each element holds the bits the verify kernel held after
// token n-1: the bits a WRITE_INTER verify stores as intermediate[n-1].

#include "common.cuh"
#include "ggml-backend-impl.h"
#include "ggml-cuda-gdn-chain.h"

#include <algorithm>

namespace {

constexpr int kState      = GGML_CUDA_GDN_CHAIN_COMMIT_STATE;       // rows = cols
constexpr int kMaxTokens  = GGML_CUDA_GDN_CHAIN_COMMIT_MAX_TOKENS;
constexpr int kLogWidth   = 2*kState + 1;                           // [g | k | delta]
constexpr int kThreads    = 256;
constexpr int kColLanes   = kThreads / kState;                      // 2
constexpr int kBlockCols  = 32;                                     // state columns per block
constexpr int kThreadCols = kBlockCols / kColLanes;                 // 16 per thread
constexpr int kStage      = 1 + kState + kBlockCols;                // floats staged per token
constexpr int kMaxLayers  = 64;                                     // per launch (kernel args)

static_assert(kThreads % kState == 0, "a block covers whole columns");
static_assert(kState % kBlockCols == 0, "blocks tile the columns");

struct commit_layers {
    const float * replay_log[kMaxLayers];
    const float * state_in[kMaxLayers];
    float *       state_out[kMaxLayers];
};

// Grid (kState / kBlockCols, H, layers). Thread (row, lane) owns rows `row`
// of columns col_base + lane*kThreadCols + j: a wave reads 32 consecutive rows
// of one column (one 128-byte segment) and shares one delta per column.
__global__ void __launch_bounds__(kThreads)
gdn_chain_commit_kernel(const commit_layers layers, const int H, const int n_commit) {
    const int layer    = blockIdx.z;
    const int head     = blockIdx.y;
    const int col_base = blockIdx.x * kBlockCols;
    const int row      = threadIdx.x % kState;
    const int lane     = threadIdx.x / kState;

    // Plain copies of the log values this block reads: gate, every key, and
    // the deltas of its columns.
    __shared__ float gate_s[kMaxTokens];
    __shared__ float key_s[kMaxTokens][kState];
    __shared__ float delta_s[kMaxTokens][kBlockCols];
    const float * log = layers.replay_log[layer];
    for (int i = threadIdx.x; i < n_commit*kStage; i += kThreads) {
        const int t = i / kStage;
        const int f = i % kStage;
        const float * transition = log + ((size_t) t*H + head)*kLogWidth;
        if (f == 0) {
            gate_s[t] = transition[0];
        } else if (f <= kState) {
            key_s[t][f - 1] = transition[f];
        } else {
            const int c = f - 1 - kState;
            delta_s[t][c] = transition[1 + kState + col_base + c];
        }
    }
    __syncthreads();

    // State layout [S, S, H]: element (row, col) of head h at
    // (h*S + col)*S + row, the GDN kernels' transposed layout.
    const size_t head_offset = (size_t) head*kState*kState;
    const float * in  = layers.state_in[layer]  + head_offset;
    float *       out = layers.state_out[layer] + head_offset;
    const int     col0 = col_base + lane*kThreadCols;

    float s[kThreadCols];
#pragma unroll
    for (int j = 0; j < kThreadCols; ++j) {
        s[j] = in[(size_t) (col0 + j)*kState + row];
    }
    for (int t = 0; t < n_commit; ++t) {
        const float g = gate_s[t];
        const float k = key_s[t][row];
#pragma unroll
        for (int j = 0; j < kThreadCols; ++j) {
            // gated_delta_net_cuda_grouped_cols:
            //     fmaf(k_reg[r], delta[c], g_val * state_shard[c][r])
            s[j] = fmaf(k, delta_s[t][lane*kThreadCols + j], g * s[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < kThreadCols; ++j) {
        out[(size_t) (col0 + j)*kState + row] = s[j];
    }
}

// Calling-thread count of successful commits (ggml-cuda-gdn-chain.h).
thread_local size_t g_gdn_chain_commit_count = 0;

} // namespace

extern "C" size_t ggml_backend_cuda_get_gdn_chain_commit_count(void) {
    return g_gdn_chain_commit_count;
}

extern "C" bool ggml_backend_cuda_gdn_chain_commit(
        ggml_backend_t backend,
        const ggml_cuda_gdn_chain_commit_layer * layers,
        int n_layers,
        int64_t S_v,
        int64_t H,
        int64_t n_log_tokens,
        int n_commit) {
    if (backend == nullptr || !ggml_backend_is_cuda(backend) ||
        layers == nullptr || n_layers <= 0 || S_v != kState ||
        H <= 0 || H > 65535 || n_commit < 1 || n_commit > kMaxTokens ||
        n_commit > n_log_tokens) {
        return false;
    }
    for (int i = 0; i < n_layers; ++i) {
        if (layers[i].replay_log == nullptr || layers[i].state_in == nullptr ||
            layers[i].state_out == nullptr) {
            return false;
        }
    }
    ggml_backend_cuda_context * ctx = (ggml_backend_cuda_context *) backend->context;
    cudaStream_t stream = ggml_cuda_enter_stream(ctx);
    (void) cudaGetLastError();
    for (int first = 0; first < n_layers; first += kMaxLayers) {
        const int n = std::min(kMaxLayers, n_layers - first);
        commit_layers args{};
        for (int i = 0; i < n; ++i) {
            args.replay_log[i] = layers[first + i].replay_log;
            args.state_in[i]   = layers[first + i].state_in;
            args.state_out[i]  = layers[first + i].state_out;
        }
        const dim3 grid(kState / kBlockCols, (unsigned int) H, (unsigned int) n);
        gdn_chain_commit_kernel<<<grid, kThreads, 0, stream>>>(args, (int) H, n_commit);
        if (cudaGetLastError() != cudaSuccess) {
            return false;
        }
    }
    ++g_gdn_chain_commit_count;
    return true;
}
