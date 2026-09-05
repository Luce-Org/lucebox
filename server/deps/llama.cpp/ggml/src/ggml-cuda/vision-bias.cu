#include "vision-bias.cuh"
#if defined(GGML_USE_HIP)
#include <climits>

bool ggml_hip_vision_bias_supported(const ggml_tensor * d) {
    if (!d || d->type != GGML_TYPE_BF16 || !ggml_is_contiguous(d)) return false;
    const auto w=d->src[0],x=d->src[1],b=d->src[2];
    if (!w || !x) return false;
    for (auto t : {w,x}) if (t->type != GGML_TYPE_BF16 || !ggml_is_contiguous(t) || t->ne[2]!=1 || t->ne[3]!=1) return false;
    if (b && (b->type!=GGML_TYPE_BF16 || !ggml_is_contiguous(b) || !ggml_is_vector(b) || b->ne[0]!=w->ne[1])) return false;
    return w->ne[0]>0 && w->ne[0]<=INT_MAX && w->ne[1]>0 && w->ne[1]<=INT_MAX &&
           x->ne[1]>0 && x->ne[1]<=INT_MAX && x->ne[0]==w->ne[0] &&
           d->ne[0]==w->ne[1] && d->ne[1]==x->ne[1] && d->ne[2]==1 && d->ne[3]==1;
}

void ggml_hip_vision_workspace_acquire(ggml_backend_cuda_context &ctx) {
    ggml_cuda_set_device(ctx.device);
    // A single event serializes shared workspace use across context streams.
    if (!ctx.vision_bias_handle) CUBLAS_CHECK(hipblasLtCreate(&ctx.vision_bias_handle));
    if (!ctx.vision_bias_workspace) {
        CUDA_CHECK(cudaMalloc(&ctx.vision_bias_workspace,GGML_HIP_VISION_WORKSPACE_BYTES));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.vision_bias_event,cudaEventDisableTiming));
    }
    if (ctx.vision_bias_launches || ctx.vision_av_launches) {
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(),ctx.vision_bias_event,0));
    }
}

void ggml_hip_vision_workspace_record(ggml_backend_cuda_context &ctx) {
    CUDA_CHECK(cudaEventRecord(ctx.vision_bias_event,ctx.stream()));
}

void ggml_hip_vision_bias(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
    GGML_ASSERT(ggml_hip_vision_bias_supported(dst));
    ggml_cuda_set_device(ctx.device);
    const auto stream=ctx.stream();
    constexpr size_t bytes=GGML_HIP_VISION_WORKSPACE_BYTES;
    ggml_hip_vision_workspace_acquire(ctx);
    const auto w=dst->src[0],x=dst->src[1],b=dst->src[2];
    const int64_t k=w->ne[0],m=w->ne[1],n=x->ne[1];
    hipblasLtMatmulDesc_t op=nullptr;
    hipblasLtMatrixLayout_t a=nullptr,bl=nullptr,c=nullptr;
    hipblasLtMatmulPreference_t pref=nullptr;
    CUBLAS_CHECK(hipblasLtMatmulDescCreate(&op,HIPBLAS_COMPUTE_32F,HIP_R_32F));
    const hipblasOperation_t ta=HIPBLAS_OP_T,tb=HIPBLAS_OP_N;
    const hipblasLtEpilogue_t epilogue=b ? HIPBLASLT_EPILOGUE_BIAS : HIPBLASLT_EPILOGUE_DEFAULT;
    CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
    CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_EPILOGUE,&epilogue,sizeof(epilogue)));
    if (b) CUBLAS_CHECK(hipblasLtMatmulDescSetAttribute(op,HIPBLASLT_MATMUL_DESC_BIAS_POINTER,&b->data,sizeof(b->data)));
    CUBLAS_CHECK(hipblasLtMatrixLayoutCreate(&a,HIP_R_16BF,k,m,k));
    CUBLAS_CHECK(hipblasLtMatrixLayoutCreate(&bl,HIP_R_16BF,k,n,k));
    CUBLAS_CHECK(hipblasLtMatrixLayoutCreate(&c,HIP_R_16BF,m,n,m));
    CUBLAS_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
    hipblasLtMatmulHeuristicResult_t heuristic{}; int returned=0;
    CUBLAS_CHECK(hipblasLtMatmulAlgoGetHeuristic(ctx.vision_bias_handle,op,a,bl,c,c,pref,1,&heuristic,&returned));
    GGML_ASSERT(returned==1 && "HIP vision bias: no first Lt heuristic; fallback forbidden");
    CUBLAS_CHECK(heuristic.state);
    GGML_ASSERT(heuristic.workspaceSize<=bytes);
    const float alpha=1,beta=0;
    CUBLAS_CHECK(hipblasLtMatmul(ctx.vision_bias_handle,op,&alpha,w->data,a,x->data,bl,&beta,
        dst->data,c,dst->data,c,&heuristic.algo,ctx.vision_bias_workspace,bytes,stream));
    ggml_hip_vision_workspace_record(ctx);
    ++ctx.vision_bias_launches;
    CUBLAS_CHECK(hipblasLtMatmulPreferenceDestroy(pref));
    CUBLAS_CHECK(hipblasLtMatrixLayoutDestroy(c));
    CUBLAS_CHECK(hipblasLtMatrixLayoutDestroy(bl));
    CUBLAS_CHECK(hipblasLtMatrixLayoutDestroy(a));
    CUBLAS_CHECK(hipblasLtMatmulDescDestroy(op));
}
#endif
