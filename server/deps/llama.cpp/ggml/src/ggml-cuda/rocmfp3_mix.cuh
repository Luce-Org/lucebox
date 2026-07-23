#pragma once
// Runtime decode support for GGML_TYPE_Q3_1_ROCMFP3_MIX (105): per-expert mixed
// absmax/adaptive ROCmFP3. The 14-byte block wire is identical to q3_0_rocmfpx;
// the per-expert codebook + mode + rotation flag live out-of-band (loaded from a
// sidecar into device buffers) and are attached here via a base-pointer registry,
// because the ggml to_fp16 converter signature carries no expert/codebook context.

#include "common.cuh"

extern "C" {

// Register per-expert decode side-data for one fused down-expert tensor.
//   base        : device pointer to the tensor's block data (src0->data)
//   nb02        : byte stride between experts (src0->nb[2])
//   n_experts   : number of experts (ne02)
//   out, in     : per-expert weight dims (rows, cols)
//   codebooks   : device buffer, n_experts * 2 * 8 __hip_bfloat16 (2 books x 8 levels)
//   modes       : device buffer, n_experts uint8 (0 legacy fixed, 1 adaptive 7s1c)
//   rotations   : device buffer, n_experts uint8 (1 => fold block-Hadamard on cols)
GGML_API void ggml_cuda_rocmfp3_mix_register(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks, const void * modes, const void * rotations);

GGML_API void ggml_cuda_rocmfp3_mix_unregister(const void * base);

}

// Dequantize one expert slice (k elements = out*in) starting at vx to half.
// Called from ggml_get_to_fp16_cuda for type 105; resolves the expert + codebook
// from the registry using the vx pointer.
void dequantize_rocmfp3_mix_to_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream);
