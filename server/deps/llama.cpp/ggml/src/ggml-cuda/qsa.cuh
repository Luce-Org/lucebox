#pragma once

#include "common.cuh"

// Selected-attention for the Qwen3.8-Next indexer path: the lightning indexer picks 4-key blocks
// per query, so attention only has to visit those. Needs the packed key/value layouts the graph
// builds in src[6]/src[7]; returns false for anything else, including the dense prefill case.
bool ggml_cuda_flash_attn_ext_qsa_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_qsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_qsa_decode_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_qsa_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
