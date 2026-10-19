#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#if defined(GGML_HIP_DS4V_VISION)
bool ggml_hip_vision_norm_capable(int device);
bool ggml_hip_vision_norm_supported(int device, const ggml_tensor * op);
void ggml_hip_vision_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif
void ggml_cuda_op_rms_norm_scale(ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * scale);
// Parity-test entry point: the fused RMS_NORM -> SCALE kernel on raw device pointers (x is [ncols, nrows, nchannels,
// nsamples] with the given strides in floats; dst is contiguous).
extern "C" { GGML_BACKEND_API int ggml_cuda_test_rms_norm_scale(const float * x, float * dst, int ncols, int nrows,
        int nchannels, int nsamples, int64_t stride_row, int64_t stride_channel, int64_t stride_sample, float eps,
        float post_scale, float bias, void * stream); }

// q8_out: optional, the output's q8_1 form as MMVQ quantizes it, rows padded
// to ncols_q8 (a multiple of QK8_1) with zero blocks.
void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor,
                                 void * q8_out = nullptr, int ncols_q8 = 0);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// dflash: residual ADD + RMS_NORM + MUL fusion (see norm.cu)
void ggml_cuda_op_add_rms_norm_mul_fused(ggml_backend_cuda_context & ctx, ggml_tensor * add_tensor, ggml_tensor * rms_tensor, ggml_tensor * mul_tensor);
