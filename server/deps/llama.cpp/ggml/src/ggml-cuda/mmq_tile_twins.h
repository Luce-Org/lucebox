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

// Every twin at x-tile 16, 32 or 48
// (mul_mat_q_case_xn), for calls wider than 16 columns. Defined next to the
// x16 entries above; the rows64 IQ4_XS/Q5_K/Q3_K/IQ2_S and rows32 Q3_K/IQ2_S
// twins exist only for these calls.
#define LUCE_MMQ_TWIN_XN_DECL(rows, name) \
    bool mul_mat_q_case_rows##rows##_##name##_xn(ggml_backend_cuda_context & ctx, const void * args, \
                                                  cudaStream_t stream, int mmq_x)
LUCE_MMQ_TWIN_XN_DECL(64, iq3_s);
LUCE_MMQ_TWIN_XN_DECL(64, iq3_xxs);
LUCE_MMQ_TWIN_XN_DECL(64, iq4_xs);
LUCE_MMQ_TWIN_XN_DECL(64, q5_k);
LUCE_MMQ_TWIN_XN_DECL(64, q3_k);
LUCE_MMQ_TWIN_XN_DECL(64, iq2_s);
LUCE_MMQ_TWIN_XN_DECL(32, iq4_xs);
LUCE_MMQ_TWIN_XN_DECL(32, q4_k);
LUCE_MMQ_TWIN_XN_DECL(32, q5_k);
LUCE_MMQ_TWIN_XN_DECL(32, q6_k);
LUCE_MMQ_TWIN_XN_DECL(32, q8_0);
LUCE_MMQ_TWIN_XN_DECL(32, iq3_s);
LUCE_MMQ_TWIN_XN_DECL(32, iq3_xxs);
LUCE_MMQ_TWIN_XN_DECL(32, q3_k);
LUCE_MMQ_TWIN_XN_DECL(32, iq2_s);
#undef LUCE_MMQ_TWIN_XN_DECL

// The 64-row twins through the prefetching kernel (mul_mat_q_case_xn_pf):
// global loads one phase ahead of the dot products, and k split while the
// caller holds the split-k scope (ggml_backend_cuda_set_mmq_splitk_scope).
// False, launching nothing, for a call it does not take; mmq.cu then runs
// the _xn entry.
#define LUCE_MMQ_TWIN_XN_PF_DECL(rows, name) \
    bool mul_mat_q_case_rows##rows##_##name##_xn_pf(ggml_backend_cuda_context & ctx, const void * args, \
                                                     cudaStream_t stream, int mmq_x)
LUCE_MMQ_TWIN_XN_PF_DECL(64, iq3_s);
LUCE_MMQ_TWIN_XN_PF_DECL(64, iq3_xxs);
LUCE_MMQ_TWIN_XN_PF_DECL(64, iq4_xs);
LUCE_MMQ_TWIN_XN_PF_DECL(64, q5_k);
LUCE_MMQ_TWIN_XN_PF_DECL(64, q3_k);
LUCE_MMQ_TWIN_XN_PF_DECL(64, iq2_s);
#undef LUCE_MMQ_TWIN_XN_PF_DECL
