#include "common.cuh"

bool ggml_cuda_top_k_qsa_supported(const ggml_tensor * op);

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
