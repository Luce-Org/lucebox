// dflash_layer_split_runtime.h — target-agnostic runtime types for the
// target layer-split pipeline.
//
// Hosts the small runtime pieces reused by layer-split drivers: a
// runtime-configuration struct and the activation double-buffer used to ferry
// hidden states between shards. Shared placement/load-plan/shard metadata lives
// in common/layer_split_utils.h; architecture-specific shard payloads keep their
// own weights/cache/graph types.

#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "peer_access.h"

#include <cstdint>
#include <vector>

namespace dflash::common {

// ── Runtime configuration (replaces globals) ────────────────────────

struct LayerSplitRuntimeConfig {
    int kq_stride_pad  = 32;    // KQ_MASK_PAD default; 256 when TBQ KV active
    int fa_window      = 0;  // 0 = full attention. qwen3.6 full-attn layers must see the whole context; a finite window drops the system prompt/tools -> breaks tool calls.
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
    ggml_type type = GGML_TYPE_F32;
};

struct ActivationBuffer {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * tensor = nullptr;
    ggml_backend_t backend = nullptr;
    int n_tokens = 0;
    ggml_type type = GGML_TYPE_F32;
};

inline void copy_layer_split_tensor(const ggml_tensor * src,
                                    int src_device,
                                    ggml_tensor * dst,
                                    int dst_device) {
    GGML_ASSERT(src && dst);
    GGML_ASSERT(src->type == dst->type);
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    GGML_ASSERT(ggml_are_same_stride(src, dst));
    if (src_device == dst_device) {
        ggml_backend_tensor_copy(src, dst);
        return;
    }

    // Route every cross-device transfer through the shared transport policy.
    // It uses reusable pinned host staging by default and P2P only after an
    // explicit --peer-access opt-in plus a successful capability check.
    if (copy_peer_async(dst->data, dst_device, src->data, src_device,
                        ggml_nbytes(src))) {
        return;
    }
    // Preserve a backend-generic last resort if the runtime-level transfer
    // fails (for example, pinned allocation pressure).
    thread_local std::vector<std::uint8_t> staging;
    staging.resize(ggml_nbytes(src));
    ggml_backend_tensor_get(src, staging.data(), 0, staging.size());
    ggml_backend_tensor_set(dst, staging.data(), 0, staging.size());
}

inline bool set_activation_tensor_from_f32(ggml_tensor * dst,
                                           const float * src,
                                           size_t offset,
                                           size_t elems) {
    if (!dst || !src) return false;
    if (dst->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(dst, src, offset, sizeof(float) * elems);
        return true;
    }
    if (dst->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(elems);
        ggml_fp32_to_fp16_row(src, tmp.data(), (int64_t)elems);
        ggml_backend_tensor_set(dst, tmp.data(), offset,
                                sizeof(ggml_fp16_t) * elems);
        return true;
    }
    if (dst->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> tmp(elems);
        ggml_fp32_to_bf16_row(src, tmp.data(), (int64_t)elems);
        ggml_backend_tensor_set(dst, tmp.data(), offset,
                                sizeof(ggml_bf16_t) * elems);
        return true;
    }
    return false;
}

inline void activation_pair_free(ActivationPair & p) {
    if (p.buf) { ggml_backend_buffer_free(p.buf); p.buf = nullptr; }
    if (p.ctx) { ggml_free(p.ctx); p.ctx = nullptr; }
    p.a = nullptr;
    p.b = nullptr;
    p.backend = nullptr;
    p.n_tokens = 0;
    p.type = GGML_TYPE_F32;
}

inline void activation_buffer_free(ActivationBuffer & b) {
    if (b.buf) { ggml_backend_buffer_free(b.buf); b.buf = nullptr; }
    if (b.ctx) { ggml_free(b.ctx); b.ctx = nullptr; }
    b.tensor = nullptr;
    b.backend = nullptr;
    b.n_tokens = 0;
    b.type = GGML_TYPE_F32;
}

inline bool activation_pair_init(ActivationPair & p,
                                 ggml_backend_t backend,
                                 int hidden,
                                 int n_tokens,
                                 ggml_type type = GGML_TYPE_F32) {
    activation_pair_free(p);
    if (n_tokens <= 0 || hidden <= 0) return false;
    p.backend = backend;
    p.n_tokens = n_tokens;
    p.type = type;
    ggml_init_params ip{};
    ip.mem_size = (size_t)8 * ggml_tensor_overhead() + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    p.ctx = ggml_init(ip);
    if (!p.ctx) return false;
    p.a = ggml_new_tensor_2d(p.ctx, type, hidden, n_tokens);
    p.b = ggml_new_tensor_2d(p.ctx, type, hidden, n_tokens);
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

inline bool activation_buffer_init(ActivationBuffer & b,
                                   ggml_backend_t backend,
                                   int hidden,
                                   int n_tokens,
                                   ggml_type type = GGML_TYPE_F32,
                                   const char * name = "target_split_activation") {
    activation_buffer_free(b);
    if (hidden <= 0 || n_tokens <= 0) return false;
    b.backend = backend;
    b.n_tokens = n_tokens;
    b.type = type;
    ggml_init_params ip{};
    ip.mem_size = (size_t)4 * ggml_tensor_overhead() + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    b.ctx = ggml_init(ip);
    if (!b.ctx) return false;
    b.tensor = ggml_new_tensor_2d(b.ctx, type, hidden, n_tokens);
    if (!b.tensor) {
        activation_buffer_free(b);
        return false;
    }
    ggml_set_name(b.tensor, name ? name : "target_split_activation");
    b.buf = ggml_backend_alloc_ctx_tensors(b.ctx, backend);
    if (!b.buf) {
        activation_buffer_free(b);
        return false;
    }
    return true;
}

} // namespace dflash::common
