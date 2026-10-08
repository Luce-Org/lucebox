// Qwen4ExpCache — KV + gated-delta-net state for Qwen3.8-Flash-Next.
// Hybrid cache: 12 full-attention layers own K/V, 36 linear-attention layers
// own a fixed recurrent state plus a depthwise-conv history. Shapes match the
// GGUF tensor layout.

#pragma once

#include "qwen4exp_internal.h"
#include "qwen4exp_mtp.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <vector>
#include <utility>

namespace luce::common {

// UMA graph-input ring: forward inputs in pinned host memory the iGPU reads
// over GTT (no H2D staging). Two slots rotate so the host never overwrites
// inputs a still-submitted graph may read.
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
// buffers; allocation assignments are remeasured whenever a graph is rebuilt.
struct Qwen4ExpDecodeWorkspace {
    ggml_context * ctx   = nullptr;
    ggml_gallocr_t alloc = nullptr;
    bool planned = false;

    // Stable T=1 graph state. The graph is rebuilt only when the fixed
    // attention-span bucket changes. QSA visibility, selection width and
    // pooled-key writes are runtime inputs, including block-completion steps.
    ggml_cgraph * gf = nullptr;
    ggml_tensor * inp_emb = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * ple_in = nullptr;
    ggml_tensor * kv_row = nullptr;
    ggml_tensor * logits = nullptr;
    ggml_tensor * hidden = nullptr;   // final HC residual, set when an MTP sidecar is loaded
    int64_t kv_bucket = 0;
    int64_t qsa_blocks = -1;  // -1 for dense; fixed score capacity otherwise
    ggml_tensor * qsa_visibility = nullptr;
    // I32[10]: valid count, four raw rows, destination row, four M-RoPE positions.
    ggml_tensor * qsa_params = nullptr;
    uint64_t builds = 0;      // smoke-test evidence: metadata addresses can be recycled
    uint64_t replays = 0;
    int qsa_budget = 0;
    int next_pos = -1;
    int max_ctx = 0;
    const Qwen4ExpWeights * model = nullptr;
    ggml_backend_t backend = nullptr;  // owns native captures; must outlive the workspace
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

    std::vector<int> full_layer_ids;    // size = 12
    std::vector<int> linear_layer_ids;  // size = 36

    // Full attention: F16 [head_dim, max_ctx, n_head_kv] (flash_attn_ext layout).
    std::vector<ggml_tensor *> attn_k;  // size = n_full
    std::vector<ggml_tensor *> attn_v;
    ggml_tensor * mtp_k = nullptr;      // MTP draft layer, same layout; null unless created with `mtp`
    ggml_tensor * mtp_v = nullptr;
    ggml_tensor * mtp_prev_hidden = nullptr; // one pending trunk row, for the next prefill chunk's first pair
    int mtp_prev_pos = -1;
    ggml_tensor * mtp_chain_hidden = nullptr; // draft HC residual, never read back between ranks
    ggml_tensor * mtp_chain_ids = nullptr;    // local subset indices; one readback after the chain
    int mtp_window = 0;                      // smoke/session-only draft attention window; 0=full

    // QSA indexer. indexer_raw holds every token's raw (pre-pool) key, [indexer_head_size, max_ctx] f32;
    // indexer_k holds pooled complete blocks (mean of `ratio` consecutive raw keys, normed and M-RoPE'd at the
    // block start), [indexer_head_size, ceil(max_ctx/ratio)+1] f32 (last row is decode scratch). `indexer_blocks` is the pooled prefix: blocks
    // past it are pooled from indexer_raw the next time QSA runs.
    std::vector<ggml_tensor *> indexer_k;    // size = n_full
    std::vector<ggml_tensor *> indexer_raw;  // size = n_full
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

    // Allocated once per cache: token-major snapshots after all k+1 verify
    // inputs. Fixed views let rollback enqueue device copies without allocation.
    int mtp_draft = 1;
    int spec_pos = -1, spec_tokens = 0;
    std::vector<ggml_tensor *> spec_ssm, spec_conv;
    using SpecRows = std::array<ggml_tensor *, QWEN4EXP_MTP_MAX_VERIFY>;
    std::vector<SpecRows> spec_ssm_rows, spec_conv_rows;
    ggml_tensor * spec_ple = nullptr;
    SpecRows spec_ple_rows{};
    std::array<std::vector<int32_t>, QWEN4EXP_MTP_MAX_VERIFY> spec_ple_prev;

    // First stable T=1 attention span of the sequence; later spans grow from it in 512-token steps, so a position's
    // span (and its attention numerics) does not depend on which forwards ran before it.
    int64_t kv_bucket_base = 0;

    // Pinned graph-input ring (see Qwen4ExpInputRing).
    Qwen4ExpInputRing input_ring;

    // T=1 decode workspace reuse; the verify and MTP draft graphs keep their own.
    Qwen4ExpDecodeWorkspace decode_workspace, verify_workspace, mtp_workspace;
};

// `mtp` adds the MTP draft layer's K/V and the verify rollback state (needs a loaded sidecar).
bool create_qwen4exp_cache(ggml_backend_t backend, const Qwen4ExpWeights & w,
                           int max_ctx, Qwen4ExpCache & out, bool mtp = false,
                           int mtp_draft = 1); // allocate the explicit draft cap once

void free_qwen4exp_cache(Qwen4ExpCache & c);

void clear_qwen4exp_decode_workspace(Qwen4ExpDecodeWorkspace & workspace);
void clear_qwen4exp_batched_decode_workspace(Qwen4ExpBatchedDecodeWorkspace & workspace);

// Zero the recurrent state, conv history and pooled-block prefix and reset
// cur_pos. KV is left intact: the next sequence overwrites it from position 0.
void reset_qwen4exp_state(ggml_backend_t backend, Qwen4ExpCache & c);

// Prefix-sized device copies. Live strip views remain valid until the cache is
// freed; callers must release snapshots first. No verify scratch is retained.
struct Qwen4ExpSnapshot {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> strips;
    int cur_pos = 0, indexer_blocks = 0, mtp_prev_pos = -1;
    int64_t kv_bucket_base = 0;
    std::vector<int32_t> ple_prev, tokens;
    std::vector<float> logits;
};

size_t qwen4exp_snapshot_bytes(ggml_backend_t backend, const Qwen4ExpCache & c, int tokens,
                             size_t * host_bytes = nullptr);
bool save_qwen4exp_snapshot(ggml_backend_t backend, const Qwen4ExpCache & c, Qwen4ExpSnapshot & s);
void restore_qwen4exp_snapshot(ggml_backend_t backend, const Qwen4ExpSnapshot & s, Qwen4ExpCache & c);
void free_qwen4exp_snapshot(Qwen4ExpSnapshot & s);

}  // namespace luce::common
