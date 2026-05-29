// Pure helpers for PFLASH_DRAFTER_SLIM tensor-skip accounting.
//
// slim_drafter_layer_bytes()  — bytes for one layer (active or dead=0)
// slim_drafter_total_bytes()  — total VRAM for a given fwd_limit + skip_output flag
//
// Used both for unit tests (no GPU required) and by load_qwen3_drafter_model()
// to report expected vs actual allocation.
#pragma once

#include <cstdint>

namespace dflash::common {

struct SlimDrafterConfig {
    int   n_embd      = 1024;
    int   n_ff        = 3072;
    int   n_head      = 16;
    int   n_head_kv   = 8;
    int   head_dim    = 128;
    int   n_vocab     = 151936;
    int   wtype_bytes = 2;    // BF16=2, Q8_0 treated as 2 for this estimate
    bool  has_qk_norm = true;
};

// Bytes for one transformer layer when active (all tensors allocated).
// When !active returns 0 (dead layers are not allocated).
inline int64_t slim_drafter_layer_bytes(const SlimDrafterConfig & c, bool active) {
    if (!active) return 0;
    const int64_t q_dim  = (int64_t)c.n_head    * c.head_dim;
    const int64_t kv_dim = (int64_t)c.n_head_kv * c.head_dim;
    int64_t b = 0;
    b += (int64_t)c.n_embd * 4;                    // attn_norm F32
    b += (int64_t)c.n_embd * q_dim  * c.wtype_bytes; // wq
    b += (int64_t)c.n_embd * kv_dim * c.wtype_bytes; // wk
    b += (int64_t)c.n_embd * kv_dim * c.wtype_bytes; // wv
    b += q_dim * (int64_t)c.n_embd  * c.wtype_bytes; // wo
    if (c.has_qk_norm) {
        b += (int64_t)c.head_dim * 4;  // q_norm F32
        b += (int64_t)c.head_dim * 4;  // k_norm F32
    }
    b += (int64_t)c.n_embd * 4;                       // ffn_norm F32
    b += (int64_t)c.n_embd * c.n_ff * c.wtype_bytes;  // ffn_gate
    b += (int64_t)c.n_embd * c.n_ff * c.wtype_bytes;  // ffn_up
    b += (int64_t)c.n_ff   * c.n_embd * c.wtype_bytes;// ffn_down
    return b;
}

// Total model VRAM estimate.
//   n_layer    — total layers in the model file
//   fwd_limit  — how many layers to actually allocate (0..n_layer)
//   skip_output — if true, output.weight (lm_head) is not allocated
inline int64_t slim_drafter_total_bytes(const SlimDrafterConfig & c,
                                        int n_layer,
                                        int fwd_limit,
                                        bool skip_output) {
    int64_t b = 0;
    b += (int64_t)c.n_embd * c.n_vocab * c.wtype_bytes; // tok_embd
    b += (int64_t)c.n_embd * 4;                          // out_norm F32
    if (!skip_output)
        b += (int64_t)c.n_embd * c.n_vocab * c.wtype_bytes; // output.weight (lm_head)
    (void)n_layer;  // kept for documentation; active layers are [0, fwd_limit)
    for (int il = 0; il < fwd_limit; ++il)
        b += slim_drafter_layer_bytes(c, /*active=*/true);
    return b;
}

} // namespace dflash::common
