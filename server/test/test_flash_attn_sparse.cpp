#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../src/pflash_ggml_adapter.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>

// Compare dense FA output vs sparse FA output
// At alpha=1.0 (select all blocks), sparse should match dense exactly.
static bool test_sparse_matches_dense(ggml_backend_t backend, int S, int H, int Hk, int D) {
    // Use no_alloc=true so tensors are NOT pre-allocated in CPU memory.
    // The gallocr will allocate them in the CUDA backend buffer instead,
    // which is required for ggml_backend_tensor_set/get to work.
    const size_t ctx_size = 256 * 1024 * 1024;
    ggml_init_params params = { ctx_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    // Q must be F32: the CUDA FA kernel asserts Q->type == GGML_TYPE_F32
    // K and V can be F16; the kernel converts them internally if needed
    // ggml FA convention: ne[0]=D, ne[1]=S, ne[2]=H
    ggml_tensor * Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, S, H);
    // K [D, S, Hk]
    ggml_tensor * K = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, S, Hk);
    // V [D, S, Hk]
    ggml_tensor * V = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, S, Hk);

    // Mark Q, K, V as graph inputs so gallocr allocates persistent backend buffers for them
    ggml_set_input(Q);
    ggml_set_input(K);
    ggml_set_input(V);

    // Causal mask for dense FA: ne[0]=KV_len, ne[1]=Q_len.
    // The kernel indexes it as mask[q * ne[0] + kv], so mask[q][kv] = (kv <= q) ? 0 : -inf.
    // pFlash applies causal masking at block granularity, so we give dense FA the same mask.
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, S, S);
    ggml_set_input(mask);

    // Dense FA
    ggml_tensor * dense_out = ggml_flash_attn_ext(ctx, Q, K, V, mask, 1.0f/sqrtf((float)D), 0.0f, 0.0f);

    // Sparse FA (alpha=1.0 = select all blocks = should match dense)
    ggml_tensor * sparse_out = ggml_flash_attn_sparse(ctx, Q, K, V, 1.0f/sqrtf((float)D), 1.0f);

    // Mark outputs so gallocr never frees/overwrites them before readback
    ggml_set_output(dense_out);
    ggml_set_output(sparse_out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dense_out);
    ggml_build_forward_expand(gf, sparse_out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    // Fill Q (F32), K (F16), V (F16) with random data
    srand(42);

    std::vector<float> q_buf(D * S * H);
    for (auto & x : q_buf) x = (float)(rand() % 1000 - 500) / 500.0f;
    ggml_backend_tensor_set(Q, q_buf.data(), 0, ggml_nbytes(Q));

    std::vector<ggml_fp16_t> buf(D * S * Hk);
    for (auto & x : buf) x = ggml_fp32_to_fp16((float)(rand() % 1000 - 500) / 500.0f);
    ggml_backend_tensor_set(K, buf.data(), 0, ggml_nbytes(K));

    buf.resize(D * S * Hk);
    for (auto & x : buf) x = ggml_fp32_to_fp16((float)(rand() % 1000 - 500) / 500.0f);
    ggml_backend_tensor_set(V, buf.data(), 0, ggml_nbytes(V));

    // Fill causal mask: mask[q * S + kv] = (kv <= q) ? 0.0f : -INFINITY
    {
        std::vector<ggml_fp16_t> mask_data(S * S);
        for (int q = 0; q < S; q++) {
            for (int kv = 0; kv < S; kv++) {
                float val = (kv <= q) ? 0.0f : -INFINITY;
                mask_data[q * S + kv] = ggml_fp32_to_fp16(val);
            }
        }
        ggml_backend_tensor_set(mask, mask_data.data(), 0, S * S * sizeof(ggml_fp16_t));
    }

    ggml_backend_graph_compute(backend, gf);

    // Compare outputs (dense_out is GGML_TYPE_F32, use ggml_nelements for element count)
    const size_t n_elems   = ggml_nelements(dense_out);
    const size_t out_bytes = n_elems * sizeof(float);
    std::vector<float> dense_data(n_elems);
    std::vector<float> sparse_data(n_elems);
    ggml_backend_tensor_get(dense_out, dense_data.data(), 0, out_bytes);
    ggml_backend_tensor_get(sparse_out, sparse_data.data(), 0, out_bytes);

    float max_diff = 0.0f;
    bool any_nonfinite = false;
    for (size_t i = 0; i < dense_data.size(); i++) {
        if (!std::isfinite(sparse_data[i]) || !std::isfinite(dense_data[i])) {
            any_nonfinite = true;
            break;
        }
        float diff = fabsf(dense_data[i] - sparse_data[i]);
        if (diff > max_diff) max_diff = diff;
    }

    const bool pass = max_diff < 1e-3f && !any_nonfinite;
    printf("[test] S=%d H=%d Hk=%d D=%d max_diff=%.6f nonfinite=%s %s\n",
           S, H, Hk, D, max_diff,
           any_nonfinite ? "YES" : "no",
           pass ? "PASS" : "FAIL");

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return pass;
}

// Sanity-check sparse attention at alpha < 1.0:
// The output should not be all zeros (basic liveness check).
// With alpha < 1.0 outputs will differ from dense FA — that is expected and not tested here.
static bool test_sparse_alpha(ggml_backend_t backend, int S, int H, int Hk, int D, float alpha) {
    const size_t ctx_size = 256 * 1024 * 1024;
    ggml_init_params params = { ctx_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    // ggml FA convention: ne[0]=D, ne[1]=S, ne[2]=H
    ggml_tensor * Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, S, H);
    ggml_tensor * K = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, S, Hk);
    ggml_tensor * V = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, S, Hk);

    ggml_set_input(Q);
    ggml_set_input(K);
    ggml_set_input(V);

    ggml_tensor * sparse_out = ggml_flash_attn_sparse(ctx, Q, K, V, 1.0f/sqrtf((float)D), alpha);
    ggml_set_output(sparse_out);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sparse_out);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(alloc, gf);

    srand(42);

    std::vector<float> q_buf(D * S * H);
    for (auto & x : q_buf) x = (float)(rand() % 1000 - 500) / 500.0f;
    ggml_backend_tensor_set(Q, q_buf.data(), 0, ggml_nbytes(Q));

    std::vector<ggml_fp16_t> buf(D * S * Hk);
    for (auto & x : buf) x = ggml_fp32_to_fp16((float)(rand() % 1000 - 500) / 500.0f);
    ggml_backend_tensor_set(K, buf.data(), 0, ggml_nbytes(K));

    buf.resize(D * S * Hk);
    for (auto & x : buf) x = ggml_fp32_to_fp16((float)(rand() % 1000 - 500) / 500.0f);
    ggml_backend_tensor_set(V, buf.data(), 0, ggml_nbytes(V));

    ggml_backend_graph_compute(backend, gf);

    const size_t n_elems   = ggml_nelements(sparse_out);
    const size_t out_bytes = n_elems * sizeof(float);
    std::vector<float> out_data(n_elems);
    ggml_backend_tensor_get(sparse_out, out_data.data(), 0, out_bytes);

    // Basic sanity: output must not be all zeros
    float max_abs = 0.0f;
    for (size_t i = 0; i < out_data.size(); i++) {
        float v = fabsf(out_data[i]);
        if (v > max_abs) max_abs = v;
    }

    bool pass = max_abs > 1e-6f;
    printf("[test_sparse_alpha] alpha=%.2f S=%d H=%d Hk=%d D=%d max_abs=%.6f %s\n",
           alpha, S, H, Hk, D, max_abs, pass ? "PASS" : "FAIL (all zeros)");

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return pass;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        fprintf(stderr, "CUDA backend not available\n");
        return 1;
    }

    pflash_register_ggml_kernel();

    bool ok = true;
    ok &= test_sparse_matches_dense(backend, 256, 16, 8, 128);   // small
    ok &= test_sparse_matches_dense(backend, 1024, 16, 8, 128);  // medium
    ok &= test_sparse_matches_dense(backend, 4096, 16, 8, 128);  // large

    // Alpha < 1.0: pFlash kernel with moderate and aggressive sparsity
    ok &= test_sparse_alpha(backend, 1024, 16, 8, 128, 0.5f);   // moderate sparsity
    ok &= test_sparse_alpha(backend, 4096, 16, 8, 128, 0.12f);  // aggressive sparsity (default alpha)

    ggml_backend_free(backend);
    printf("\n%s\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return ok ? 0 : 1;
}
