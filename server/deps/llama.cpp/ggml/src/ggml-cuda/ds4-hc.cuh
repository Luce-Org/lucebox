#pragma once

#include "ggml-cuda/common.cuh"

// Fused DeepSeek4 hyper-connection (HC) decode ops. See ds4-hc.cu for the
// per-mode contract (pre / post / out). Used by the opt-in
// DFLASH_DS4_FUSED_DECODE single-graph decode path; output is deterministic
// but not bit-identical to the CPU HC reference (expf ULP differences
// amplified by the sinkhorn iterations).
void ggml_cuda_op_ds4_hc(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
