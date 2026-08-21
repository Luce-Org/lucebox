// SpecLA factor lifecycle helpers (docs/SPECLA.md).
//
// The first entry is the immediate DeltaConstruct compatibility path used by
// the fully factorized verifier:
//
//   S_l ← exp(g⁺_A) · S_l + Σ_{t ∈ path} exp(g⁺_A − g⁺_t) · k_t ⊗ ṽ_t
//
// The ggml-graph implementation of the same math needs ~6 small ops per layer
// (48 layers ⇒ hundreds of kernel launches ⇒ ~5 ms of launch overhead per
// commit on ROCm without HIP graphs). The production HLD path uses the compact
// and final-flush helpers below; its normal accepted update is delayed into
// the next verification kernel.
//
// Compiled for CUDA and (via the hip_compat <cuda_runtime.h> shim with
// LANGUAGE HIP) for ROCm — same shared-.cu pattern as
// geometric_draft_topk_cuda.cu.

#pragma once

#include <cstddef>
#include <cstdint>

namespace dflash::common {

// Consolidated double-bank factor pointers. Bank 0 and bank 1 alternate
// between "pending" (consumed by the next verify / final flush) and "current"
// (receiving the just-run verify's factors).
struct SpeclaFactorBanks {
    float * k[2]    = {nullptr, nullptr};
    float * v[2]    = {nullptr, nullptr};
    float * g[2]    = {nullptr, nullptr};
    float * conv[2] = {nullptr, nullptr};
};

// ssm_ptrs_dev: DEVICE array of n_delta pointers, one per delta layer's
//               [S_k, S_v, H] f32 state tensor (ne0 = k-dim rows).
// fk/fv/fg:     consolidated factor buffers, f32, token axis outermost:
//               fk [S_k, H, n_delta, T], fv [S_v, H, n_delta, T],
//               fg [H, n_delta, T].
// idx_dev:      DEVICE array of A accepted token indices, deepest last.
// stream:       cudaStream_t / hipStream_t (nullptr = default stream).
// launched:     set once the kernel has been accepted by the runtime. A false
//               return with launched=true is not safe to retry in place.
bool specla_commit_fused(float * const * ssm_ptrs_dev,
                         const float * fk,
                         const float * fv,
                         const float * fg,
                         const int32_t * idx_dev,
                         int A, int S_k, int S_v, int H, int n_delta,
                         void * stream,
                         bool * launched = nullptr);

// Compact an accepted tree path from arbitrary flat-node indices into
// contiguous token slots in the alternate bank. All delta layers and both
// recurrent factor families are copied by one kernel launch.
bool specla_compact_fused(const float * src_k,
                          const float * src_v,
                          const float * src_g,
                          const float * src_conv,
                          float * dst_k,
                          float * dst_v,
                          float * dst_g,
                          float * dst_conv,
                          const int32_t * idx_dev,
                          int A, int S_k, int S_v, int H,
                          int n_delta, int conv_channels,
                          void * stream);

// Materialize a final delayed path when generation ends before another verify
// can consume it. Factors are raw per-token recurrence terms in path order.
bool specla_flush_raw_fused(float * const * ssm_ptrs_dev,
                            float * const * conv_ptrs_dev,
                            const float * fk,
                            const float * fv,
                            const float * fg,
                            const float * conv,
                            int A, int S_k, int S_v, int H,
                            int n_delta, int conv_channels, int d_conv,
                            void * stream);

// Apply an accepted prefix from a raw token-major convolution factor bank to
// every durable per-layer convolution window. `conv` is [C, L, T]; each
// state pointer addresses a contiguous [d_conv-1, C] tensor.
bool specla_commit_conv_raw_fused(float * const * conv_ptrs_dev,
                                  const float * conv,
                                  int A, int n_tokens,
                                  int n_delta, int conv_channels,
                                  int d_conv, void * stream);

// Shared bank-lifecycle helpers used by both the production Qwen35 target and
// the test_dflash harness. `pending_bank` is the bank the just-run verify
// consumed; the opposite bank received that verify's factors.
//
// Rotate the just-verified factors into the pending role. A pure chain
// acceptance is already contiguous and only switches banks. A tree acceptance
// that walked a sibling is compacted from the current bank into the old
// pending bank in accepted-path order via specla_compact_fused; `idx_dev`
// must already hold the `commit_n` accepted DFS indices. On success the new
// pending bank is written to `out_pending_bank`.
bool specla_rotate_pending_factors(const SpeclaFactorBanks & banks,
                                   const int32_t * idx_dev,
                                   int pending_bank,
                                   bool walked_sibling,
                                   int commit_n,
                                   int S_k, int S_v, int H,
                                   int n_delta, int conv_channels,
                                   void * stream,
                                   int * out_pending_bank);

// Apply the pending bank's raw factors to the durable SSM/conv states. Used at
// generation boundaries where no next verify will consume the pending path.
bool specla_flush_pending_factors(const SpeclaFactorBanks & banks,
                                  float * const * ssm_ptrs_dev,
                                  float * const * conv_ptrs_dev,
                                  int pending_bank,
                                  int pending_count,
                                  int S_k, int S_v, int H,
                                  int n_delta, int conv_channels, int d_conv,
                                  void * stream);

}  // namespace dflash::common
