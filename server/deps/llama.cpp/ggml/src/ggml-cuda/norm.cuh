#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#if defined(GGML_USE_HIP)
bool ggml_hip_vision_norm_capable(int device);
bool ggml_hip_vision_norm_supported(int device, const ggml_tensor * op);
void ggml_hip_vision_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// dflash: residual ADD + RMS_NORM + MUL fusion (see norm.cu)
void ggml_cuda_op_add_rms_norm_mul_fused(ggml_backend_cuda_context & ctx, ggml_tensor * add_tensor, ggml_tensor * rms_tensor, ggml_tensor * mul_tensor);
