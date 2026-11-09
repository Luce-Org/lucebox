// GPU top-K + log-prob extraction for DDTree draft distributions.
//
// Drop-in device-side replacement for extract_draft_topk (ddtree.cpp): instead
// of D2H-ing the full [vocab × n_positions] draft logits and running an OpenMP
// heap top-K + online-logsumexp on the CPU, this runs the whole thing on the
// GPU directly on the logits tensor's device buffer and copies back only the
// [n_positions × K] results.
//
// Semantics match extract_draft_topk exactly:
//   scaled        = logit * (1 / max(1e-3, temperature))
//   log_z         = logsumexp_j(scaled_j)                       (per position)
//   out_log_probs = top-K scaled logits (desc), minus log_z
//   out_token_ids = matching vocab ids; ties broken toward the lower id
//
// Returns false (caller must fall back to the CPU path) when the GPU runtime is
// unavailable, the pointer is not device memory, K is out of range, or any
// device call fails. Compiled on both CUDA and HIP/ROCm builds — this same .cu
// is compiled directly with LANGUAGE HIP on ROCm, the cuda_runtime.h spellings
// mapped by the hip_compat shim; guarded by LUCE_HAVE_DRAFT_TOPK. See
// CMakeLists.txt.

#pragma once

#include <cstdint>

namespace luce::common {

inline constexpr bool geometric_draft_topk_cuda_supports_k(int K) noexcept {
    return (K >= 1 && K <= 8) || K == 12 || K == 16;
}

// d_logits: device pointer to row-major [n_positions][vocab] f32 logits (the
//           position stride is `vocab` floats — pass an offset pointer to skip
//           leading positions). out_* are HOST buffers of size n_positions*K.
bool geometric_extract_draft_topk_cuda(const void * d_logits,
                             int n_positions, int vocab, int K,
                             float * out_log_probs,
                             int32_t * out_token_ids,
                             float temperature);

// The device that owns `d_logits` when it is device memory (the condition
// under which geometric_extract_draft_topk_cuda runs on the GPU), else -1.
// A caller can check a stable buffer once instead of on every call.
int geometric_draft_topk_cuda_device(const void * d_logits);

// geometric_extract_draft_topk_cuda queued on `stream` (a cudaStream_t of
// `device`, passed as void *): the same split and kernels, so the same
// results, with the token ids written to the device buffer d_out_token_ids
// and the log-probs copied to out_log_probs (host memory; pinned keeps the
// copy asynchronous). Nothing is synchronized: read the outputs only after
// the stream ran. Returns false, and the caller takes the CPU path, where
// the synchronous call would (K, scratch, or a launch error).
bool geometric_extract_draft_topk_cuda_async(const void * d_logits,
                             int device,
                             int n_positions, int vocab, int K,
                             float temperature,
                             void * stream,
                             int32_t * d_out_token_ids,
                             float * out_log_probs);

}  // namespace luce::common
