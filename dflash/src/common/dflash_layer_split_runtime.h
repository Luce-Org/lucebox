// dflash_layer_split_runtime.h — target-agnostic runtime types for the
// DFlash layer-split pipeline.
//
// Hosts the small pieces that are reused by every architecture's layer-split
// driver: a runtime-configuration struct (replaces former globals) and an
// activation double-buffer used to ferry hidden states between shards.
// Architecture-specific shard layouts (e.g. qwen35's TargetLayerSplitShard
// that embeds TargetWeights/TargetCache) live in their own headers.

#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

namespace dflash27b {

// ── Runtime configuration (replaces globals) ────────────────────────

struct LayerSplitRuntimeConfig {
    int kq_stride_pad  = 32;    // KQ_MASK_PAD default; 256 when TBQ KV active
    int fa_window      = 2048;  // flash-attn sliding window
    int draft_ctx_max  = 4096;  // draft context cap
    int draft_swa_window = 0;   // draft SWA window (0 = disabled)
};

// ── Activation double-buffer for inter-shard transfer ───────────────

struct ActivationPair {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;
    ggml_backend_t backend = nullptr;
    int n_tokens = 0;
};

inline void activation_pair_free(ActivationPair & p) {
    if (p.buf) { ggml_backend_buffer_free(p.buf); p.buf = nullptr; }
    if (p.ctx) { ggml_free(p.ctx); p.ctx = nullptr; }
    p.a = nullptr;
    p.b = nullptr;
    p.backend = nullptr;
    p.n_tokens = 0;
}

inline bool activation_pair_init(ActivationPair & p,
                                 ggml_backend_t backend,
                                 int hidden,
                                 int n_tokens) {
    activation_pair_free(p);
    if (n_tokens <= 0 || hidden <= 0) return false;
    p.backend = backend;
    p.n_tokens = n_tokens;
    ggml_init_params ip{};
    ip.mem_size = (size_t)8 * ggml_tensor_overhead() + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    p.ctx = ggml_init(ip);
    if (!p.ctx) return false;
    p.a = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, hidden, n_tokens);
    p.b = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, hidden, n_tokens);
    if (!p.a || !p.b) {
        activation_pair_free(p);
        return false;
    }
    ggml_set_name(p.a, "target_split_act_a");
    ggml_set_name(p.b, "target_split_act_b");
    p.buf = ggml_backend_alloc_ctx_tensors(p.ctx, backend);
    if (!p.buf) {
        activation_pair_free(p);
        return false;
    }
    return true;
}

} // namespace dflash27b
