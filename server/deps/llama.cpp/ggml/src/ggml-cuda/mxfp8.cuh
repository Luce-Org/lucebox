#pragma once
#include "common.cuh"

// MXFP8 (native E4M3 weights, E8M0 scale per 32) times F32 activations, F32 output: the
// weights are decoded exactly in registers and the activations are never quantized, so the
// result is the BF16-dense arithmetic at one byte per weight. Up to MXFP8_MMV_MAX_NCOLS columns.
// Each column's sum has the same operation order for every column count (batch invariant).
#define MXFP8_MMV_MAX_NCOLS 8

bool ggml_cuda_mxfp8_mul_mat_vec_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
void ggml_cuda_mxfp8_mul_mat_vec(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, cudaStream_t stream);
