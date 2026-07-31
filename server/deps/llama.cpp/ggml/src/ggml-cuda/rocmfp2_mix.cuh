#pragma once
// Runtime decode support for GGML_TYPE_Q2_1_ROCMFP2_MIX (106): per-expert mixed
// absmax/adaptive ROCmFP2. The 10-byte block wire is identical to q2_0_rocmfpx, so
// a GGUF splice from qtype 107 is offset-preserving;
// the per-expert codebook + mode + rotation flag live out-of-band (loaded from a
// sidecar into device buffers) and are attached here via a base-pointer registry,
// because the ggml to_fp16 converter signature carries no expert/codebook context.

#include "common.cuh"

extern "C" {

// Register per-expert decode side-data for one gate or up expert tensor.
// NOTE: the serving GGUF stores gate and up SEPARATELY (ffn_gate_exps /
// ffn_up_exps), and one codebook pair is fitted per fused expert covering
// BOTH halves -- so the two tensors register the SAME codebook buffer. Do not
// assume a registered base owns its codebook exclusively.
//   base        : device pointer to the tensor's block data (src0->data)
//   nb02        : byte stride between experts (src0->nb[2])
//   n_experts   : number of experts (ne02)
//   out, in     : per-expert weight dims (rows, cols)
//   codebooks   : device buffer, n_experts * 2 * 4 __hip_bfloat16 (2 books x 4 levels)
//   modes       : device buffer, n_experts uint8 (0 fixed qtype-107 levels, 1 adaptive 7s1c)
//   rotations   : device buffer, n_experts uint8 -- always 0 here. The qtype-106
//                 encoder never emits rotation and the kernel ignores it; the
//                 loader passes zeros and rejects anything else.
GGML_API void ggml_cuda_rocmfp2_mix_register(
        const void * base, size_t nb02, int n_experts, int out, int in,
        const void * codebooks, const void * modes, const void * rotations);

GGML_API void ggml_cuda_rocmfp2_mix_unregister(const void * base);

}

// Dequantize one expert slice (k elements = out*in) starting at vx to half.
// Called from ggml_get_to_fp16_cuda for type 106; resolves the expert + codebook
// from the registry using the vx pointer.
void dequantize_rocmfp2_mix_to_fp16_cuda(const void * vx, half * y, int64_t k, cudaStream_t stream);

// Fused quantized matvec for type 106 (the MMVQ-style decode path): computes
// y[out, ncols] = W[out,in] . x[in, ncols] by decoding the quantized blocks
// inline (per-expert codebook/mode from the registry) instead of the
// dequantize->cuBLAS round-trip. vx is the expert slice base (registry lookup
// resolves expert+codebook+mode). x/y are f32; *_col_stride are element strides
// between columns (tokens). Used for batch-1 decode; larger batches keep the
// dequant->cuBLAS fallback. Returns true if it handled the op.
bool ggml_cuda_rocmfp2_mix_mul_mat_vec(
        const void * vx, const float * x, float * y,
        int in, int out, int ncols,
        int64_t x_col_stride, int64_t y_col_stride, cudaStream_t stream);


// Fused DENSE 3-D-slice matvec: dst[out, ntokens, nslices] = W[out, in, nslices] .
// x[in, ntokens, nslices], one slice per blockIdx.y. Exists because the target's
// attn_output_a is mul_mat'd as a 3-D batched slice (src1->ne[2] > 1), which the 2-D
// hook rejects, and the dequant->cuBLAS fallback reads more bytes than the f16 it
// replaces. Requires the tensor registered with n_experts >= nslices; returns false
// otherwise (rather than reading another tensor's codebook).
// Strides are named by MEANING deliberately: the underlying kernel's _s1/_s2 are the
// SLOT and TOKEN strides respectively, which is the reverse of what ne[1]/ne[2] suggests.
bool ggml_cuda_rocmfp2_mix_mul_mat_vec_3d(
        const void * vx, const float * src1, float * dst,
        int in, int out, int nslices, int ntokens,
        int64_t src1_token_stride, int64_t src1_slice_stride,
        int64_t dst_token_stride,  int64_t dst_slice_stride,
        cudaStream_t stream);

// Stream-sync-free fused MoE matvec for a qtype-106 mul_mat_id (decode). Reads
// the routing `ids` on device so the whole op runs with no host id-sort +
// cudaStreamSynchronize (the generic ggml_cuda_mul_mat_id fallback needs both),
// which also lets the decode FFN subgraph be CUDA-graph captured. Bit-identical
// per output element to the fallback's per-expert-slice path. vx is the whole
// MoE tensor base (registry resolves per-expert codebook/mode on device). All
// *_s* args are element strides. Returns false if vx is not registered.
// Fused gate/up + SwiGLU_DS4 in ONE launch. Collapses the second mul_mat_id and the separate
// swiglu pass that qtype 106 was paying and qtype 107 was not (the backend already fuses the
// trio for types mmvq can serve). Returns false unless both halves are registered and their
// shapes match, leaving the two-launch path untouched.
bool ggml_cuda_rocmfp2_mix_mul_mat_id_glu(
        const void * vx_up, const void * vx_gate,
        const float * src1, const int32_t * ids, float * dst,
        int in, int out, int n_expert_used, int n_tokens, int ne11,
        int64_t ids_s0, int64_t ids_s1,
        int64_t src1_s1, int64_t src1_s2,
        int64_t dst_s1, int64_t dst_s2,
        float glu_limit, cudaStream_t stream);

bool ggml_cuda_rocmfp2_mix_mul_mat_id(
        const void * vx, const float * src1, const int32_t * ids, float * dst,
        int in, int out, int n_expert_used, int n_tokens, int ne11,
        int64_t ids_s0, int64_t ids_s1,
        int64_t src1_s1, int64_t src1_s2,
        int64_t dst_s1, int64_t dst_s2, cudaStream_t stream);

// True if a qtype-106 tensor base is registered (used by the graph-usability
// check to confirm the sync-free mul_mat_id path will handle the node).
bool ggml_cuda_rocmfp2_mix_registered(const void * vx);
