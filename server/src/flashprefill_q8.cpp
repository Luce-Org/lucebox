// ggml flash_attn_ext-based FlashPrefill implementation.
//
// Provides flash_prefill_forward_q8() — a portable alternative to the custom
// BF16 WMMA flashprefill kernels. Uses ggml's built-in flash_attn_ext. The
// ggml CUDA/HIP FA kernels require F32 Q/output while still supporting half
// K/V, so this path widens Q and the temporary attention output inside the
// graph and copies the result back to the caller's half/F32 output buffer.
//
// The attention is run in chunks (CHUNK_S tokens per pass) to keep the
// causal mask small and the ggml graph allocation bounded. Each chunk
// attends to all K/V positions up to the end of that chunk (full causal).

#include "flashprefill.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace dflash::common {
namespace flashprefill {

namespace {
constexpr int CHUNK_S = 4096;
}

int flash_prefill_forward_q8(
    ggml_backend_t backend,
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    ggml_type qkv_type,
    const FlashPrefillConfig & cfg)
{
    (void)cfg;  // block-sparse selection not used in this path

    const int B  = batch;
    (void)B;  // used only for shape validation in the future
    const int S  = seq_len;
    const int H  = n_q_heads;
    const int Hk = n_k_heads;
    const int D  = head_dim;

    if (qkv_type != GGML_TYPE_F16 && qkv_type != GGML_TYPE_BF16
        && qkv_type != GGML_TYPE_F32) {
        std::fprintf(stderr, "[flashprefill_q8] unsupported qkv_type=%s\n",
                     ggml_type_name(qkv_type));
        return -1;
    }
    const int elem_size = (int)ggml_type_size(qkv_type);

    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));

    for (int cs = 0; cs < S; cs += CHUNK_S) {
        const int cl     = std::min(CHUNK_S, S - cs);
        const int kv_len = cs + cl;

        // ── Build ggml graph for this chunk ──
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 40
                      + ggml_graph_overhead_custom(256, false)
                      + 64 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            std::fprintf(stderr, "[flashprefill_q8] ggml_init failed\n");
            ggml_gallocr_free(galloc);
            return -1;
        }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 256, false);

        // External tensors wrapping the caller's raw pointers.
        // Q_buf layout: [D, H, S] contiguous (the drafter's persistent buf).
        // K/V layout:   [D, Hk, S] contiguous.
        const size_t esz = (size_t)elem_size;

        ggml_tensor * Q_full = ggml_new_tensor_3d(ctx, qkv_type, D, H, S);
        Q_full->data = const_cast<void *>(Q);
        ggml_set_name(Q_full, "Q_ext");
        ggml_set_input(Q_full);

        ggml_tensor * K_full = ggml_new_tensor_3d(ctx, qkv_type, D, Hk, S);
        K_full->data = const_cast<void *>(K);
        ggml_set_name(K_full, "K_ext");
        ggml_set_input(K_full);

        ggml_tensor * V_full = ggml_new_tensor_3d(ctx, qkv_type, D, Hk, S);
        V_full->data = const_cast<void *>(V);
        ggml_set_name(V_full, "V_ext");
        ggml_set_input(V_full);

        ggml_tensor * O_full = ggml_new_tensor_3d(ctx, qkv_type, D, H, S);
        O_full->data = O;
        ggml_set_name(O_full, "O_ext");
        ggml_set_output(O_full);

        // Q chunk: [D, H, cl] view, then permute -> [D, cl, H] for FA.
        // ggml-cuda/hip flash_attn_ext requires F32 Q.
        ggml_tensor * Q_chunk = ggml_view_3d(ctx, Q_full, D, H, cl,
                                             esz * D, esz * D * H,
                                             (size_t)cs * esz * D * H);
        ggml_tensor * Q_perm = ggml_cont(ctx, ggml_permute(ctx, Q_chunk, 0, 2, 1, 3));
        ggml_tensor * Q_fa = Q_perm;
        if (qkv_type != GGML_TYPE_F32) {
            Q_fa = ggml_cpy(ctx, Q_perm,
                            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, cl, H));
        }

        // K/V: [D, Hk, kv_len] view, then permute → [D, kv_len, Hk] for FA
        ggml_tensor * K_view = ggml_view_3d(ctx, K_full, D, Hk, kv_len,
                                            esz * D, esz * D * Hk, 0);
        ggml_tensor * V_view = ggml_view_3d(ctx, V_full, D, Hk, kv_len,
                                            esz * D, esz * D * Hk, 0);
        ggml_tensor * K_fa = ggml_cont(ctx, ggml_permute(ctx, K_view, 0, 2, 1, 3));
        ggml_tensor * V_fa = ggml_cont(ctx, ggml_permute(ctx, V_view, 0, 2, 1, 3));

        // Causal mask: [kv_len, cl] f16
        ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_len, cl);
        ggml_set_name(mask, "causal_mask");
        ggml_set_input(mask);

        ggml_tensor * attn = ggml_flash_attn_ext(ctx, Q_fa, K_fa, V_fa,
                                                 mask, scale, 0.0f, 0.0f);
        // FA output is F32 [D, H, cl] (permuted). Copy/cast to O_full at
        // chunk offset so the caller can keep compact half persistent buffers.
        ggml_tensor * O_dst = ggml_view_3d(ctx, O_full, D, H, cl,
                                           esz * D, esz * D * H,
                                           (size_t)cs * esz * D * H);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, attn, O_dst));

        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            std::fprintf(stderr, "[flashprefill_q8] graph alloc failed at cs=%d\n", cs);
            ggml_free(ctx);
            ggml_gallocr_free(galloc);
            return -1;
        }

        // Fill causal mask
        {
            std::vector<uint16_t> mask_data((size_t)kv_len * cl);
            const uint16_t zero_f16 = 0;
            const uint16_t ninf_f16 = 0xFC00;
            for (int q_local = 0; q_local < cl; ++q_local) {
                int q_global = cs + q_local;
                for (int k = 0; k < kv_len; ++k) {
                    mask_data[(size_t)q_local * kv_len + k] =
                        (k <= q_global) ? zero_f16 : ninf_f16;
                }
            }
            ggml_backend_tensor_set(mask, mask_data.data(), 0,
                                    (size_t)kv_len * cl * sizeof(uint16_t));
        }

        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        ggml_free(ctx);
    }

    ggml_gallocr_free(galloc);
    return 0;
}

} // namespace flashprefill
} // namespace dflash::common
