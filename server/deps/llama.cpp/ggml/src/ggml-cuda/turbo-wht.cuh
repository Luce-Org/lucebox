#pragma once

#include "common.cuh"

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Fused FWHT-rotate + quantize: reads (possibly strided) F32 src, applies
// tq3_rotate_forward, and writes quantized output (Q4_0 or Q8_0).
// Eliminates the intermediate F32 buffer from separate turbo_wht + cpy.
void ggml_cuda_op_turbo_wht_quantize(ggml_backend_cuda_context & ctx,
                                      const ggml_tensor * src, ggml_tensor * dst,
                                      ggml_type quant_type);
