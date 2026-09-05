#include "vision-av.cuh"
#include "vision-bias.cuh"
#include "norm.cuh"

#if defined(GGML_USE_HIP)
bool ggml_hip_vision_av_f32_supported(int device, const ggml_tensor * op) {
    if (!ggml_hip_vision_norm_capable(device) || !op || op->op != GGML_OP_MUL_MAT_VISION_AV_F32 ||
        !op->src[0] || !op->src[1] || op->type != GGML_TYPE_F32) {
        return false;
    }
    for (int i = 2; i < GGML_MAX_SRC; ++i) {
        if (op->src[i]) return false;
    }
    const auto v = op->src[0], p = op->src[1];
    if (v->type != GGML_TYPE_F32 || p->type != GGML_TYPE_F32 ||
        v->ne[0] != 64 || v->ne[1] != 16 || v->ne[3] != 1) {
        return false;
    }
    const int64_t n = v->ne[2];
    return n >= 16 && n <= 4096 && p->ne[0] == n && p->ne[1] == n && p->ne[2] == 16 && p->ne[3] == 1 &&
        op->ne[0] == 64 && op->ne[1] == n && op->ne[2] == 16 && op->ne[3] == 1 &&
        ggml_is_contiguous(v) && ggml_is_contiguous(p) && ggml_is_contiguous(op);
}

void ggml_hip_vision_av_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_hip_vision_av_f32_supported(ctx.device, dst));
    ggml_cuda_set_device(ctx.device);
    const auto stream = ctx.stream();
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    GGML_ASSERT(capture == cudaStreamCaptureStatusNone && "HIP vision AV requires direct execution");
    ggml_hip_vision_workspace_acquire(ctx);
    const int64_t n = dst->src[0]->ne[2];
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    CUBLAS_CHECK(hipblasLtMatmulDescCreate(&op,HIPBLAS_COMPUTE_32F,HIP_R_32F));
    const hipblasOperation_t transpose = HIPBLAS_OP_N;
    CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
    CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_TRANSB,&transpose,sizeof(transpose)));
    // Pinned Torch 3d3aa833 Blas.cpp/CUDABlas.cpp: row-major output swaps the
    // operands, retaining V's token-major stride. Default epilogue, C equals D.
    CUBLAS_CHECK(hipblasLtMatrixLayoutCreate(&a,HIP_R_32F,64,n,1024));
    CUBLAS_CHECK(hipblasLtMatrixLayoutCreate(&b,HIP_R_32F,n,n,n));
    CUBLAS_CHECK(hipblasLtMatrixLayoutCreate(&c,HIP_R_32F,64,n,64));
    const int batches = 16;
    const int64_t stride_a = 64, stride_b = n*n, stride_c = n*64;
    for (auto layout : {a,b,c}) {
        CUBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(layout,HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batches,sizeof(batches)));
    }
    CUBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(a,HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&stride_a,sizeof(stride_a)));
    CUBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(b,HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&stride_b,sizeof(stride_b)));
    CUBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(c,HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&stride_c,sizeof(stride_c)));
    CUBLAS_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    constexpr size_t bytes = GGML_HIP_VISION_WORKSPACE_BYTES;
    CUBLAS_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
    hipblasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    CUBLAS_CHECK(hipblasLtMatmulAlgoGetHeuristic(ctx.vision_bias_handle,op,a,b,c,c,pref,1,&heuristic,&returned));
    GGML_ASSERT(returned == 1 && "HIP vision AV: no first Lt heuristic; fallback forbidden");
    CUBLAS_CHECK(heuristic.state);
    GGML_ASSERT(heuristic.workspaceSize <= bytes);
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(hipblasLtMatmul(ctx.vision_bias_handle,op,&alpha,dst->src[0]->data,a,dst->src[1]->data,b,
        &beta,dst->data,c,dst->data,c,&heuristic.algo,ctx.vision_bias_workspace,bytes,stream));
    ggml_hip_vision_workspace_record(ctx);
    ++ctx.vision_av_launches;
    CUBLAS_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
    CUBLAS_CHECK(hipblasLtMatrixLayoutDestroy(c));
    CUBLAS_CHECK(hipblasLtMatrixLayoutDestroy(b));
    CUBLAS_CHECK(hipblasLtMatrixLayoutDestroy(a));
    CUBLAS_CHECK(hipblasLtMatmulDescDestroy(op));
}
#endif
