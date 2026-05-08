#include "flashprefill.h"

// Forward-declare the registration function from ggml-cuda (defined in fattn-sparse.cu).
// No extern "C" — nvcc compiles .cu as C++ and the symbol has C++ linkage.
void ggml_cuda_flash_attn_sparse_set_kernel(
    int (*fn)(const void*, const void*, const void*, void*,
              int, int, int, int, int, float, float));

static int pflash_adapter(
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale, float alpha)
{
    dflash27b::flashprefill::FlashPrefillConfig cfg;
    if (alpha >= 1.0f) {
        // alpha >= 1.0 means "select all blocks" — configure for dense attention
        cfg.alpha          = 0.0f;
        cfg.attention_sink = seq_len;  // all blocks are "sinks"
        cfg.window         = seq_len;  // window covers everything
        cfg.last_n_full    = seq_len;  // all query blocks attend fully
    } else {
        cfg.alpha = alpha;
    }
    return dflash27b::flashprefill::flash_prefill_forward_bf16(
        Q, K, V, O,
        batch, seq_len, n_q_heads, n_k_heads, head_dim,
        scale, cfg);
}

// Call this once at init time before running any ggml_flash_attn_sparse graphs.
void pflash_register_ggml_kernel() {
    ggml_cuda_flash_attn_sparse_set_kernel(&pflash_adapter);
}
