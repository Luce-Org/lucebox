// Qwen4ExpCache — KV + gated-delta-net state for Qwen3.8-Flash-Next.
// Hybrid cache: 12 full-attention layers own K/V, 36 linear-attention layers
// own a fixed recurrent state plus a depthwise-conv history. Shapes match the
// GGUF tensor layout.

#pragma once

#include "qwen4exp_internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <vector>

namespace luce::common {

// UMA graph-input ring: forward inputs in pinned host memory the iGPU reads
// over GTT (no H2D staging). Two slots rotate so the host never overwrites
// inputs a still-submitted graph may read. Opt out: LUCE_HIP_NO_UMA_RING=1.
struct Qwen4ExpInputRing {
    bool                  enabled    = false;
    int                   next_slot  = 0;
    uint64_t              writes     = 0;      // slots handed out since enable
    ggml_backend_buffer_t buf        = nullptr;
    char *                base       = nullptr;
    size_t                slot_bytes = 0;
    // Per-slot section layout: [ embd | positions | ple | mask ].
    size_t                embd_off = 0, pos_off = 0, ple_off = 0, mask_off = 0;
    size_t                embd_cap = 0, pos_cap = 0, ple_cap = 0, mask_cap = 0;
};

// Optional T=1 decode workspace. Reuse the metadata arena and gallocr backing
// buffers; allocation assignments are remeasured because KV views and RoPE
// positions advance every step, so the graph still has to be rebuilt.
struct Qwen4ExpDecodeWorkspace {
    ggml_context * ctx   = nullptr;
    ggml_gallocr_t alloc = nullptr;
    bool planned = false;

    // Stable T=1 graph state (QWEN4EXP_DECODE_STABLEGRAPH=1). The graph is
    // rebuilt only when the fixed attention-span bucket changes.
    ggml_cgraph * gf = nullptr;
    ggml_tensor * inp_emb = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * ple_in = nullptr;
    ggml_tensor * kv_row = nullptr;
    ggml_tensor * logits = nullptr;
    int64_t kv_bucket = 0;
    uint64_t stable_calls = 0;
};

// Shared arena for exact-width independent-sequence decode. Unlike the stable
// single-slot graph, the graph is rebuilt for each call because its state
// tensor edges depend on the active slot ordering. The metadata arena and
// allocator backing storage are still shared across calls.
struct Qwen4ExpBatchedDecodeWorkspace {
    ggml_context * ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    bool planned = false;
};

struct Qwen4ExpCache {
    ggml_context *        ctx     = nullptr;
    ggml_backend_buffer_t buf     = nullptr;

    int       max_ctx  = 0;
    int       cur_pos  = 0;
    ggml_type kv_type  = GGML_TYPE_F16;

    std::vector<int> full_layer_ids;    // size = 12
    std::vector<int> linear_layer_ids;  // size = 36

    // Full attention: [head_dim, max_ctx, n_head_kv] (flash_attn_ext layout).
    std::vector<ggml_tensor *> attn_k;  // size = n_full
    std::vector<ggml_tensor *> attn_v;

    // QSA indexer keys, normed + M-RoPE'd at absolute block positions:
    // [indexer_head_size, ceil(max_ctx/ratio)] f32. `indexer_blocks` is the
    // populated prefix QSA selection must never read past.
    std::vector<ggml_tensor *> indexer_k;  // size = n_full
    int indexer_blocks = 0;

    // Gated delta net: ssm_state [S_v, S_v, H_v] f32;
    //                  conv_state [kernel-1, conv_channels] f32.
    std::vector<ggml_tensor *> ssm_state;   // size = n_linear
    std::vector<ggml_tensor *> conv_state;

    // Per-layer embedding (PLE) conv history, one per PLE layer:
    // [ple_hist, hc_dim] f32 where ple_hist = (ple_conv_kernel-1)*ple_ngram_size.
    std::vector<ggml_tensor *> ple_conv_state;
    std::vector<int> ple_layer_ids;

    // Rolling window of the last (ple_ngram_size - 1) token ids, oldest first,
    // for the host-side PLE n-gram hash across decode steps.
    std::vector<int32_t> ple_prev;

    // Pinned graph-input ring (see Qwen4ExpInputRing).
    Qwen4ExpInputRing input_ring;

    // T=1 decode workspace reuse, on by default (disable with
    // QWEN4EXP_DECODE_REUSE=0; excluded under QWEN4EXP_UPSTREAM=1).
    Qwen4ExpDecodeWorkspace decode_workspace;
};

bool create_qwen4exp_cache(ggml_backend_t backend, const Qwen4ExpWeights & w,
                           int max_ctx, ggml_type kv_type, Qwen4ExpCache & out);

void free_qwen4exp_cache(Qwen4ExpCache & c);

void clear_qwen4exp_decode_workspace(Qwen4ExpDecodeWorkspace & workspace);
void clear_qwen4exp_batched_decode_workspace(Qwen4ExpBatchedDecodeWorkspace & workspace);

// Zero the recurrent state and conv history and reset cur_pos. KV is left
// intact; callers that need a clean sequence also reset cur_pos themselves.
void reset_qwen4exp_state(ggml_backend_t backend, Qwen4ExpCache & c);

}  // namespace luce::common
