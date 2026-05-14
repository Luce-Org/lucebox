// Smoke test: create a Gemma4 KV cache with TQ3_0 quantization and validate
// the resulting cache structure, alignment, and layer-to-KV-index mappings.
//
// Usage: test_gemma4_kv_tq3 <gemma4.gguf>

#include "internal.h"
#include "gemma4.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
#  define setenv(name, value, overwrite) _putenv_s(name, value)
#endif

using namespace dflash27b;

static void fail(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <gemma4.gguf>\n", argv[0]);
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    GemmaTargetWeights w;
    if (!load_gemma4_target_gguf(argv[1], backend, w)) {
        std::fprintf(stderr, "load_gemma4_target_gguf: %s\n", dflash27b_last_error());
        ggml_backend_free(backend);
        return 1;
    }
    std::printf("[target] n_layer=%d n_embd=%d n_head_kv=%d head_dim=%d "
                "n_layer_kv=%d n_capture_layers=%d\n",
                w.n_layer, w.n_embd, w.n_head_kv, w.head_dim,
                w.n_layer_kv, w.n_capture_layers);

    // Set KV type environment variables to tq3_0 before cache creation
    setenv("DFLASH27B_KV_K", "tq3_0", 1);
    setenv("DFLASH27B_KV_V", "tq3_0", 1);

    const int max_ctx = 1024;
    GemmaTargetCache cache;
    if (!create_gemma4_cache(w, max_ctx, backend, cache)) {
        std::fprintf(stderr, "create_gemma4_cache: %s\n", dflash27b_last_error());
        free_gemma4_target_weights(w);
        ggml_backend_free(backend);
        return 1;
    }
    std::printf("[cache] created max_ctx=%d kv_slots=%zu\n",
                cache.max_ctx, cache.attn_k.size());

    // Assert KV types resolved correctly
    if (cache.kv_k_type != GGML_TYPE_TQ3_0) {
        char buf[64];
        std::snprintf(buf, sizeof(buf),
            "kv_k_type=%s expected tq3_0", ggml_type_name(cache.kv_k_type));
        fail(buf);
    }
    if (cache.kv_v_type != GGML_TYPE_TQ3_0) {
        char buf[64];
        std::snprintf(buf, sizeof(buf),
            "kv_v_type=%s expected tq3_0", ggml_type_name(cache.kv_v_type));
        fail(buf);
    }
    std::printf("[types] kv_k=%s kv_v=%s OK\n",
                ggml_type_name(cache.kv_k_type),
                ggml_type_name(cache.kv_v_type));

    // Validate layer_to_kv_idx mapping
    if ((int)cache.layer_to_kv_idx.size() != w.n_layer) {
        char buf[64];
        std::snprintf(buf, sizeof(buf),
            "layer_to_kv_idx.size()=%zu expected %d",
            cache.layer_to_kv_idx.size(), w.n_layer);
        fail(buf);
    }

    const int n_kv_slots = (int)cache.attn_k.size();
    int n_shared_layers = 0;
    for (int il = 0; il < w.n_layer; il++) {
        const int idx = cache.layer_to_kv_idx[il];
        if (idx == -1) {
            n_shared_layers++;
        } else if (idx < 0 || idx >= n_kv_slots) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "layer_to_kv_idx[%d]=%d out of range [0, %d)",
                il, idx, n_kv_slots);
            fail(buf);
        }
    }
    std::printf("[kv_idx] n_kv_slots=%d n_shared_layers=%d n_layer=%d\n",
                n_kv_slots, n_shared_layers, w.n_layer);

    // Validate layer_to_donor_kv: shared layers must have a valid donor
    if ((int)cache.layer_to_donor_kv.size() != w.n_layer) {
        fail("layer_to_donor_kv.size() != n_layer");
    }
    for (int il = 0; il < w.n_layer; il++) {
        if (cache.layer_to_kv_idx[il] == -1) {
            // This is a shared layer — must have a valid donor
            const int donor = cache.layer_to_donor_kv[il];
            if (donor < 0 || donor >= n_kv_slots) {
                char buf[128];
                std::snprintf(buf, sizeof(buf),
                    "layer_to_donor_kv[%d]=%d invalid for shared layer (n_kv_slots=%d)",
                    il, donor, n_kv_slots);
                fail(buf);
            }
        }
    }
    std::printf("[donor_kv] all shared layers have valid donors OK\n");

    // Validate TQ3_0 alignment: for TQ3_0, KV tensors must have ne[1] % 256 == 0
    // (create_gemma4_cache rounds max_ctx_alloc up to a multiple of 256 for TQ3_0).
    for (int i = 0; i < n_kv_slots; i++) {
        const ggml_tensor * K = cache.attn_k[i];
        const ggml_tensor * V = cache.attn_v[i];
        if (!K) { char buf[32]; std::snprintf(buf, sizeof(buf), "attn_k[%d] is null", i); fail(buf); }
        if (!V) { char buf[32]; std::snprintf(buf, sizeof(buf), "attn_v[%d] is null", i); fail(buf); }
        if (K->ne[1] % 256 != 0) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "attn_k[%d]: ne[1]=%" PRId64 " not a multiple of 256 (TQ3_0 alignment)",
                i, K->ne[1]);
            fail(buf);
        }
        if (V->ne[1] % 256 != 0) {
            char buf[128];
            std::snprintf(buf, sizeof(buf),
                "attn_v[%d]: ne[1]=%" PRId64 " not a multiple of 256 (TQ3_0 alignment)",
                i, V->ne[1]);
            fail(buf);
        }
    }
    std::printf("[alignment] all %d KV tensors are 256-aligned OK\n", n_kv_slots);

    // Validate target_feat tensor
    if (!cache.target_feat) fail("target_feat is null");
    const int64_t expected_feat_ne0 = (int64_t)w.n_capture_layers * w.n_embd;
    if (cache.target_feat->ne[0] != expected_feat_ne0) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "target_feat->ne[0]=%" PRId64 " expected %" PRId64
            " (n_capture_layers=%d * n_embd=%d)",
            cache.target_feat->ne[0], expected_feat_ne0,
            w.n_capture_layers, w.n_embd);
        fail(buf);
    }
    std::printf("[target_feat] ne=[%" PRId64 ", %" PRId64 "] type=%s OK\n",
                cache.target_feat->ne[0], cache.target_feat->ne[1],
                ggml_type_name(cache.target_feat->type));

    // Print cache stats
    std::printf("[stats] n_kv_slots=%d max_ctx=%d kv_seq_dim=%" PRId64
                " target_feat_cap=%d\n",
                n_kv_slots, cache.max_ctx,
                cache.attn_k[0]->ne[1],
                cache.target_feat_cap);

    free_gemma4_cache(cache);
    free_gemma4_target_weights(w);
    ggml_backend_free(backend);
    std::printf("PASS\n");
    return 0;
}
