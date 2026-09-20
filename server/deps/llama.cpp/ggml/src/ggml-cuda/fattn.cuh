#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
long long ggml_backend_cuda_get_fattn_qsa_launch_count();
long long ggml_backend_cuda_get_fattn_dense_launch_count();
void ggml_backend_cuda_reset_fattn_launch_counts();

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);
