// Bridges into the big-tile (128x128, 8-warp) MMQ instances that coexist with
// the GGML_CUDA_MMQ_SMALL_TILE (64x64, 4-warp) default instances on RDNA.
//
// The tile shape is baked into every device/host constexpr in mmq.cuh via
// macros, so one TU can only hold one shape. The *-big.cu template instances
// re-include mmq.cuh inside `namespace lucebox_mmq_big` with no tile macro
// defined (the upstream 128x128 RDNA default), which gives the second shape
// distinct symbols. `args` is the caller's ::mmq_args passed as void const *:
// the namespaced struct is textually identical (its layout does not depend on
// the tile macros), the bridge casts it back.
//
// Why: at spec-decode verify widths (N <= 32) the 64-row tile measured
// +12-23% (grid occupancy on a 64-CU gfx1201), but at prefill widths the
// narrow x-tile re-streams the weights (+12% MMQ time at N = 512). The
// runtime dispatch in mmq.cu picks per shape and keeps both wins.
#pragma once

#include "common.cuh"

// Defined in template-instances/mmq-instance-<type>-big.cu.
void mul_mat_q_case_big_iq4_xs(ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);
void mul_mat_q_case_big_q4_k  (ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);
void mul_mat_q_case_big_q5_k  (ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);
void mul_mat_q_case_big_q6_k  (ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);
void mul_mat_q_case_big_q8_0  (ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);
