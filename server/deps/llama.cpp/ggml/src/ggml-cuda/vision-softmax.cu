#include "vision-softmax.cuh"

#if defined(GGML_USE_HIP)
#include "vision-softmax-kernels.cuh"
#include <climits>

bool ggml_hip_vision_softmax_f32_supported(int device, const ggml_tensor * op) {
    const auto & info = ggml_cuda_info();
    if (device < 0 || device >= info.device_count || info.devices[device].warp_size != 32 ||
        !op || op->op != GGML_OP_SOFT_MAX_VISION_F32 || !op->src[0] ||
        op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
        op->ne[0] < 16 || op->ne[0] > 4096) {
        return false;
    }
    int64_t elements = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (op->ne[i] != op->src[0]->ne[i] || op->ne[i] <= 0 ||
            op->ne[i] > (INT_MAX / int64_t(sizeof(float))) / elements) {
            return false;
        }
        elements *= op->ne[i];
    }
    if (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) ||
        op->view_offs % 16 != 0 || op->src[0]->view_offs % 16 != 0 ||
        uintptr_t(op->data) % 16 != 0 || uintptr_t(op->src[0]->data) % 16 != 0) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_SRC; ++i) {
        if (op->src[i]) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_op_soft_max_vision_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_hip_vision_softmax_f32_supported(ctx.device, dst));
    const int width = int(dst->ne[0]);
    const int rows = int(ggml_nelements(dst) / width);
    const auto * input = static_cast<const float *>(dst->src[0]->data);
    auto * output = static_cast<float *>(dst->data);
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture));
    GGML_ASSERT(capture == cudaStreamCaptureStatusNone);
    // A contiguous Torch allocation has an aligned base; row misalignment is
    // reproduced inside the non-vector-width branch of the kernel.
    GGML_ASSERT(uintptr_t(input) % 16 == 0 && uintptr_t(output) % 16 == 0);
    ggml_vision_softmax::launch(input, output, width, rows, ctx.stream());
    CUDA_CHECK(cudaGetLastError());
    ++ctx.vision_softmax_launches;
}
#endif
