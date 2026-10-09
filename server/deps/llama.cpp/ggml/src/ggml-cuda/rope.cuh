#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

// RMS_NORM * w -> forward NORMAL/TAIL ROPE -> F16 copy in one launch (see
// rope.cu); writes the ROPE node's F32 output and the copy's F16 output.
bool ggml_cuda_rms_norm_rope_f16_supported(int ncols);
void ggml_cuda_rms_norm_rope_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * norm,
                                 const ggml_tensor * mul, const ggml_tensor * rope, const ggml_tensor * cpy);

// A forward NORMAL/TAIL rope of a contiguous [D, H, T] F32 tensor that also
// writes the q8_1 form MMVQ quantizes its grouped view [D * hpg, T, H / hpg]
// to (see rope.cu).
void ggml_cuda_rope_q8(ggml_backend_cuda_context & ctx, const ggml_tensor * rope, void * q8, int hpg);
// ROPE (multi-section, F32) -> PERMUTE -> CONT in one kernel: the rotation is written straight into cont's layout.
void ggml_cuda_op_rope_permuted(ggml_backend_cuda_context & ctx, ggml_tensor * rope, ggml_tensor * cont);
