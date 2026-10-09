#pragma once

#include "ggml-cuda/common.cuh"

// Fused DeepSeek4 hyper-connection (HC) decode ops. See ds4-hc.cu for the
// per-mode contract (pre / post / out). Used by the opt-in
// LUCE_DS4_FUSED_DECODE single-graph decode path; output is deterministic
// but not bit-identical to the CPU HC reference (expf ULP differences
// amplified by the sinkhorn iterations).
void ggml_cuda_op_ds4_hc(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// One HC sub-block boundary (rms_norm, fn mix, Sinkhorn pre, post, collapse,
// rms_norm * w) per token in one launch; see ds4-hc.cu. mmv_block is the MMVF
// block size the unfused fn product would launch with (it fixes the
// partial-sum order).
bool ggml_cuda_ds4_hc_boundary_supported(int n_hc, int n_embd, int mmv_block);
void ggml_cuda_ds4_hc_boundary(
        ggml_backend_cuda_context & ctx, int n_tokens, int n_hc, int mmv_block,
        const float * R, int64_t R_stride, const half * fn, int64_t fn_stride, const float * base,
        const float * block_out, int64_t bo_stride, const float * w,
        float * pre_dst, int64_t pre_stride, float * R_next, int64_t Rn_stride,
        float * out, int64_t out_stride,
        int n_embd, float eps_hc, float eps_out, int iters,
        float pre_scale, float post_scale, float comb_scale, bool write_pre,
        void * q8_out);   // optional: out quantized to q8_1 as MMVQ would (rows padded to MATRIX_ROW_PADDING)

// The DS4 router's select (mode 4) and weights (mode 5) ops on the same
// logits in one launch, bit-identical to the two.
void ggml_cuda_op_ds4_router_fused(ggml_backend_cuda_context & ctx, ggml_tensor * select, ggml_tensor * weights);
