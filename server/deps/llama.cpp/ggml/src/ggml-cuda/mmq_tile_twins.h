// Bridges into the MMQ tile twins: narrower row tiles for MMQ calls of up to
// 48 columns on RDNA4. mmq.cu picks one and its x-tile per call
// (mmq-tile-twin-gate.h).
//
// Same construction as mmq_big.h: template-instances/mmq-instance-<type>-
// rows<rows>.cu re-includes mmq.cuh inside its own namespace with that
// shape's tile macros, which gives the twin distinct symbols next to the
// parent instance. Each twin has one entry, mul_mat_q_case_twin at x-tile 16,
// 32 or 48: calls of 16 or fewer columns take x-tile 16, the one every
// instance picks there. Only the tiled launch is compiled, with device code
// for RDNA4 targets only (stream-k and its fixup kernel are left out, RDNA4
// never uses them). The 64-row twins run plain 2D calls of 17-48 columns
// through the prefetching kernel, which splits k while the caller holds the
// split-k scope (ggml_backend_cuda_set_mmq_splitk_scope).
//
//   rows64  64-row / 4-warp tile
//   rows32  32-row / 2-warp tile   (outputs <= 6144 rows;
//            GGML_CUDA_MMQ_SMALL_TILE_ROWS 32)
//
// `args` is the caller's ::mmq_args passed as void const *; the namespaced
// struct is textually identical, and each twin copies the bytes into its own
// type before reading them. A twin returns false, launching nothing, when it
// cannot serve the call (a non-HIP build, an x-tile it does not compile, a
// stream-k call); the caller then runs the parent instance.
#pragma once

#include "common.cuh"

// Defined in template-instances/mmq-instance-<name>-rows<rows>.cu.
#define LUCE_MMQ_TWIN_DECL(rows, name) \
    bool mul_mat_q_case_rows##rows##_##name(ggml_backend_cuda_context & ctx, const void * args, \
                                             cudaStream_t stream, int mmq_x)
LUCE_MMQ_TWIN_DECL(64, iq3_s);
LUCE_MMQ_TWIN_DECL(64, iq3_xxs);
LUCE_MMQ_TWIN_DECL(64, iq4_xs);
LUCE_MMQ_TWIN_DECL(64, q5_k);
LUCE_MMQ_TWIN_DECL(64, q3_k);
LUCE_MMQ_TWIN_DECL(64, iq2_s);
LUCE_MMQ_TWIN_DECL(32, iq4_xs);
LUCE_MMQ_TWIN_DECL(32, q4_k);
LUCE_MMQ_TWIN_DECL(32, q5_k);
LUCE_MMQ_TWIN_DECL(32, q6_k);
LUCE_MMQ_TWIN_DECL(32, q8_0);
LUCE_MMQ_TWIN_DECL(32, iq3_s);
LUCE_MMQ_TWIN_DECL(32, iq3_xxs);
LUCE_MMQ_TWIN_DECL(32, q3_k);
LUCE_MMQ_TWIN_DECL(32, iq2_s);
#undef LUCE_MMQ_TWIN_DECL
