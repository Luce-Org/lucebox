// Bridges into the MMQ tile twins: narrower row tiles for MMQ calls
// of 16 or fewer columns on RDNA4. mmq.cu picks one per call
// (mmq-tile-twin-gate.h).
//
// Same construction as mmq_big.h: template-instances/mmq-instance-<type>-
// rows<rows>.cu re-includes mmq.cuh inside its own namespace with that
// shape's tile macros, which gives the twin distinct symbols next to the
// parent instance. Each twin compiles only the 16-column x-tile
// (mul_mat_q_case_x16), the one every instance picks at 16 or fewer columns,
// and only its tiled launch: mul_mat_q<type, 16, need_check false/true>, two
// kernels, with device code for RDNA4 targets only (stream-k and its fixup
// kernel are left out, RDNA4 never uses them).
//
//   rows64  64-row / 4-warp tile
//   rows32  32-row / 2-warp tile   (outputs <= 6144 rows;
//            GGML_CUDA_MMQ_SMALL_TILE_ROWS 32)
//
// `args` is the caller's ::mmq_args passed as void const *; the namespaced
// struct is textually identical, and each twin copies the bytes into its own
// type before reading them. A twin returns false, launching nothing,
// when it cannot serve the call (a non-HIP build, a call wider than 16
// columns, a stream-k call); the caller then runs the parent instance.
#pragma once

#include "common.cuh"

// Defined in template-instances/mmq-instance-<type>-rows64.cu.
bool mul_mat_q_case_rows64_iq3_s  (ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);
bool mul_mat_q_case_rows64_iq3_xxs(ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);

// Defined in template-instances/mmq-instance-<type>-rows32.cu.
bool mul_mat_q_case_rows32_iq4_xs (ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);
bool mul_mat_q_case_rows32_q4_k   (ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);
bool mul_mat_q_case_rows32_q5_k   (ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);
bool mul_mat_q_case_rows32_q6_k   (ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);
bool mul_mat_q_case_rows32_q8_0   (ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);
bool mul_mat_q_case_rows32_iq3_s  (ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);
bool mul_mat_q_case_rows32_iq3_xxs(ggml_backend_cuda_context & ctx, const void * args, cudaStream_t stream);
