#pragma once

// SINGLE_REPLAY (server src/qwen35/gdn_single_commit.h):
// the single-sequence chain verify runs its GATED_DELTA_NET as a mapped verify
// (no final state, no per-token states, replay log recorded), and rollback
// commits the accepted prefix from the replay log with the verify kernel's
// own update. Nothing here runs unless the server calls it; the GDN op and the
// paged replay-log commit (gdn-replay-log.cu) are unchanged.
//
// A separate header so that adding these declarations does not rebuild every
// ggml-cuda translation unit (common.cuh includes ggml-cuda.h).

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Head size the commit supports: the grouped-cols kernel's (S_v = 128).
#define GGML_CUDA_GDN_CHAIN_COMMIT_STATE 128
// Most transitions one commit applies (a block-16 chain verify).
#define GGML_CUDA_GDN_CHAIN_COMMIT_MAX_TOKENS 16

// Whether a non-KDA, non-tree GATED_DELTA_NET with head size S_v that records
// a replay log runs the grouped-cols kernel on `device`: the route
// launch_gated_delta_net (gated_delta_net.cu) picks, including a route a
// test pinned with ggml_backend_cuda_set_gdn_route_override, read now.
// Only that kernel updates the state as fmaf(k, delta, g * state), which is
// what ggml_backend_cuda_gdn_chain_commit computes; the scalar kernel's
// g * state + k * delta may contract the other way. Host-only, no device call.
GGML_BACKEND_API bool ggml_backend_cuda_gdn_chain_uses_grouped_cols(int device, int64_t S_v);

// One delta layer of a chain commit. All device pointers on the backend's
// device, F32, contiguous:
//   replay_log  [2*S_v + 1, H, n_log_tokens, 1]: per token and head
//               [g | k[S_v] | delta[S_v]] as the GDN kernel recorded them
//               (ggml_gated_delta_net_capture_replay_log's layout)
//   state_in    [S_v, S_v, H]: the state the verify started from
//   state_out   [S_v, S_v, H]: receives the committed state; may equal
//               state_in (each element is read once, then written, by the
//               same thread). Must not overlap any other layer's buffers.
struct ggml_cuda_gdn_chain_commit_layer {
    const float * replay_log;
    const float * state_in;
    float       * state_out;
};

// state_out = the state after the first n_commit tokens of the replay log,
// for every layer: per element (row, col) of head h, starting from state_in,
//     s = fmaf(k_t[row], delta_t[col], g_t * s)    for t = 0 .. n_commit-1,
// the grouped-cols GDN kernel's update with its own k, delta and g values.
// Asynchronous on the backend's compute stream (stream-ordered after its
// graph computes); one launch per 64 layers; no host sync, no device query.
// S_v must be GGML_CUDA_GDN_CHAIN_COMMIT_STATE and
// 1 <= n_commit <= min(n_log_tokens, GGML_CUDA_GDN_CHAIN_COMMIT_MAX_TOKENS).
// Returns false when an argument is invalid (nothing launched) or a launch
// failed (layers of earlier launches may have been committed).
GGML_BACKEND_API bool ggml_backend_cuda_gdn_chain_commit(
        ggml_backend_t backend,
        const struct ggml_cuda_gdn_chain_commit_layer * layers,
        int n_layers,
        int64_t S_v,
        int64_t H,
        int64_t n_log_tokens,
        int n_commit);

// Calling-thread count of ggml_backend_cuda_gdn_chain_commit calls that
// returned true. Tests use it to check that a rollback took the commit.
GGML_BACKEND_API size_t ggml_backend_cuda_get_gdn_chain_commit_count(void);

#ifdef __cplusplus
}
#endif
