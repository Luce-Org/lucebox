// Shared inputs/outputs for the DFlash draft graph builder.
#pragma once

#include "ggml.h"

#include <vector>

namespace dflash::common {

struct DraftWeights; // fwd

struct DraftGraphInputs {
    int           ctx_len;          // length of target_hidden_cat along ne[1]
    ggml_tensor * noise_embed;      // [hidden, q_len=16, 1] f32
    ggml_tensor * target_hidden_cat;// [5*hidden, ctx_len, 1] f32
    ggml_tensor * positions_q;      // [q_len] i32   values [ctx_len..ctx_len+q_len-1]
    ggml_tensor * positions_k;      // [ctx_len+q_len] i32   values [0..ctx_len+q_len-1]
    // Optional: if non-null, the graph projects final hidden states through
    // this LM head (shape [hidden, vocab]) and returns logits instead of
    // hidden states. Used for DFlash integration where the draft shares the
    // target's lm_head.
    ggml_tensor * lm_head;
    // Optional: causal mask for SWA layers [kv_pad, q_len] F16.
    // nullptr = all layers non-causal.
    ggml_tensor * causal_mask_swa = nullptr;
    // Optional: mask for NON-SWA layers [kv_pad, q_len] F16. Used when the
    // context is padded to a stable allocation size for CUDA-graph replay:
    // real keys are 0, pad keys are -inf. nullptr = no mask (legacy).
    ggml_tensor * pad_mask_full = nullptr;
};

struct DraftGraphOutputs {
    ggml_tensor * hidden_states;    // [hidden, q_len, 1]  (always set)
    ggml_tensor * logits;           // [vocab, q_len, 1]   (non-null iff lm_head was provided)
};

DraftGraphOutputs build_draft_graph(
    ggml_context *            ctx,
    const DraftWeights &      w,
    const DraftGraphInputs &  in);

// ── Cached drafter context-KV ────────────────────────────────────────
// The ctx side of the draft forward (feature fusion → per-layer K/V →
// k_norm → RoPE) is row-independent, so committed rows can be computed once
// and kept in per-layer ring caches; the step graph then only processes the
// q_len noise rows and flash-attends over the cache. RoPE uses ABSOLUTE
// positions (attention scores only depend on position differences, so this
// matches the legacy rebased-window math) which keeps cached entries valid
// as the feature window slides.
//
// Cache layout (head-major, mirrors the target KV cache): one row per
// position holding all KV heads fused, [head_dim*n_head_kv, kv_total] F16.
// Rows [0, cap) are the ctx ring (position p → slot p % cap), rows
// [cap, cap+q_len) are the per-step noise K/V scratch, row cap+q_len is a
// trash slot for padded append rows, and the remainder is mask alignment.

struct DraftKvCacheRefs {
    int kv_total = 0;             // cache rows incl. noise scratch + trash + pad
    std::vector<ggml_tensor *> k; // per layer [head_dim*n_head_kv, kv_total] f16
    std::vector<ggml_tensor *> v; // per layer [head_dim*n_head_kv, kv_total] f16
};

// Fuse `n_rows` feature rows and set_rows their per-layer K/V into the caches.
struct DraftKvAppendInputs {
    int           n_rows = 0;
    ggml_tensor * feat = nullptr;      // [n_target_layers*hidden, n_rows] f32
    ggml_tensor * positions = nullptr; // [n_rows] i32, absolute
    ggml_tensor * rows = nullptr;      // [n_rows] i32, destination cache slots
};
bool build_draft_kv_append(
    ggml_context *             ctx,
    ggml_cgraph *              gf,
    const DraftWeights &       w,
    const DraftKvCacheRefs &   cache,
    const DraftKvAppendInputs & in);

// One draft step over the cached context KV. Noise K/V are written into the
// scratch slots and the flash attention reads the full kv_total span; the
// masks carry window membership, causality and slot validity.
struct DraftKvStepInputs {
    ggml_tensor * noise_embed = nullptr; // [hidden, q_len] f32
    ggml_tensor * positions_q = nullptr; // [q_len] i32, absolute
    ggml_tensor * noise_rows = nullptr;  // [q_len] i32, scratch cache slots
    ggml_tensor * mask_full = nullptr;   // [kv_total, q_len] f16 (non-SWA layers)
    ggml_tensor * mask_swa = nullptr;    // [kv_total, q_len] f16 (SWA layers)
    ggml_tensor * lm_head = nullptr;     // optional fused projection
};
DraftGraphOutputs build_draft_kv_step(
    ggml_context *            ctx,
    ggml_cgraph *             gf,
    const DraftWeights &      w,
    const DraftKvCacheRefs &  cache,
    const DraftKvStepInputs & in);

} // namespace dflash::common
