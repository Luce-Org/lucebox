// Qwen4Exp forward graph.
//
// qwen4exp_forward() runs n_tokens new tokens starting at cache position pos0
// through the 48-layer hybrid trunk and writes last-token logits. The multi-slot
// batched decode entry point below handles independent one-token slot rows.
// Ported from upstream llama.cpp src/models/qwen4exp.cpp
// (hyper-connections, gated delta net linear attention, dense full attention,
// 512-expert top-10 MoE, per-layer n-gram embedding) into Luzebox's ggml graph
// style.
//
// Single sequence (n_seqs = 1). On gfx1151 and RDNA4, multi-row prompt prefill uses QSA
// with F32 accumulation, including below the selection budget. T=1 retains
// dense attention below the budget and selected attention beyond it. Verify
// rows use the T=1 attention path at each position, excluding prompt promotion.
// The PLE table is read through Qwen4ExpPleReader, never uploaded in full.

#pragma once

#include "qwen4exp_internal.h"
#include "qwen4exp_cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <vector>

namespace luce::common {

// QSA indexer block pooling: [idim, nb*r] keys -> [idim, nb], block b = mean of tokens r*b .. r*b+r-1.
ggml_tensor * qwen4exp_pool_blocks(ggml_context * c, ggml_tensor * keys, int64_t r);

// K/V span of the stable T=1 decode graph at kv_len. The sequence's first span (`base`, set on first use) leaves at
// least one 256-token window past its kv_len; later spans grow from it in 512-token steps. These are the spans a
// token-by-token decode rebuilds with, as a function of kv_len alone, so a verify row attends over the same span
// (same attention numerics) as plain decode at that position.
int64_t qwen4exp_stable_kv_span(int64_t & base, int64_t max_ctx, int64_t kv_len);

struct Qwen4ExpForwardResult {
    bool ok = false;
};

// Host-only input preparation; safe to run for the next prompt chunk while
// the current graph computes. The caller owns the immutable token span.
struct Qwen4ExpInputs {
    bool ok = false;
    std::vector<float> emb, ple;
    std::vector<int32_t> ple_prev;
};
Qwen4ExpInputs qwen4exp_prepare_inputs(const Qwen4ExpWeights & w,
    const int32_t * tokens, int n_tokens, const std::vector<int32_t> & ple_prev);

struct Qwen4ExpGraphMemory {
    size_t graph = 0, inputs = 0, mask = 0, host = 0, scratch = 0, metadata = 0;
};
// Allocation plan only: no GPU allocation, input reads, compute or state update.
Qwen4ExpGraphMemory qwen4exp_graph_memory(ggml_backend_t backend, const Qwen4ExpWeights & w,
    Qwen4ExpCache & cache, int n_tokens, int pos0, bool verify = false);
Qwen4ExpGraphMemory qwen4exp_mtp_graph_memory(ggml_backend_t backend, const Qwen4ExpWeights & w,
    Qwen4ExpCache & cache, int n_tokens, int pos0);

// One independent sequence span for batch eligibility and per-slot solo fallback.
struct Qwen4ExpForwardSegment {
    Qwen4ExpCache * cache = nullptr;
    const int32_t * tokens = nullptr;
    int n_tokens = 0;
    int pos0 = 0;
};

// Pure decision for validated spans: at most four one-token rows, all dense in the solo path.
// use_qsa is the cached gfx1151 capability.
bool qwen4exp_can_batch(const Qwen4ExpWeights & w,
                       const Qwen4ExpForwardSegment * segments, int n_segments, bool use_qsa);

// Run the trunk. `tokens` has n_tokens entries, processed as one contiguous
// single-sequence span at positions [pos0, pos0 + n_tokens). On success the
// cache is advanced to pos0 + n_tokens and out_logits holds n_vocab floats
// for the final token. out_hidden, when set, receives every token's final HC
// residual (n_embd * n_hc floats each) for the MTP draft head.
//
// verify (MTP speculation, 2 <= n_tokens <= k+1, qwen4exp_verify_supported): each
// token is computed exactly as a T=1 forward at its position computes it
// (batch-invariant matmuls, per-token attention), out_logits holds all rows,
// and the cache keeps the state after every token for
// qwen4exp_verify_rollback.
// mtp_prefill (n_tokens > 1, contiguous prompt chunks): append K/V-only MTP
// slices to this graph. out_hidden receives only the last pending trunk row.
Qwen4ExpForwardResult qwen4exp_forward(ggml_backend_t backend,
                                       const Qwen4ExpWeights & w,
                                       Qwen4ExpCache & cache,
                                       const int32_t * tokens,
                                       int n_tokens,
                                       int pos0,
                                       std::vector<float> & out_logits,
                                       std::vector<float> * out_hidden = nullptr,
                                       bool verify = false,
                                       bool mtp_prefill = false,
                                       const Qwen4ExpInputs * inputs = nullptr);

// The cache was created with `mtp` (and a loaded sidecar).
bool qwen4exp_verify_supported(const Qwen4ExpCache & cache);

// Retain the first `retained` verify inputs (accepted drafts + 1, or fewer at EOS).
// Restores recurrent/conv/PLE state and truncates KV and QSA visibility by position.
bool qwen4exp_verify_rollback(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache, int pos0, int retained = 1);

// MTP draft step over (trunk hidden h_p, token x_{p+1}) pairs at positions [pos0, pos0 + n): runs the sidecar's
// nextn projection and layer, writes the draft layer's K/V there, and returns the logits of the last pair (its
// argmax drafts x_{p+2}). `hidden` holds n rows of the trunk's final HC residual (n_embd * n_hc floats each), as
// returned by qwen4exp_forward's out_hidden. Optional out_hidden returns the
// last MTP HC residual, which feeds the next autoregressive draft step.
// kv_only fills the same prompt K/V without evaluating attention or the draft head;
// out_logits is cleared and out_hidden must be null. Slice sizes stay unchanged.
// last_only preserves every K/V write but evaluates only the final attention/FFN row.
// The default full-row/full-head path is retained as the smoke oracle.
bool qwen4exp_mtp_forward(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                          const int32_t * tokens, const float * hidden, int n, int pos0,
                          std::vector<float> & out_logits, std::vector<float> * out_hidden = nullptr, bool kv_only = false, bool last_only = false);

// Catch up pending trunk pairs, then chain k predictions with the MTP residual.
bool qwen4exp_mtp_draft(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                        const int32_t * tokens, const float * hidden, int n, int pos0, int k,
                        std::vector<int32_t> & drafts);

// Decode one next token for each independent slot. `caches[s]` owns that
// sequence's KV and recurrent state; `tokens[s]` and `positions[s]` are never
// interpreted as a common time axis. The shared workspace must outlive calls
// and is normally owned by the sequence-engine/model instance.
// If any slot needs QSA or more than four slots are active, use per-slot solo forwards.
// Reference caches also use per-slot solo forwards. The server admits at most four slots.
Qwen4ExpForwardResult qwen4exp_forward_batched(
                                       ggml_backend_t backend,
                                       const Qwen4ExpWeights & w,
                                       Qwen4ExpCache * const * caches,
                                       const int32_t * tokens,
                                       const int32_t * positions,
                                       int n_slots,
                                       Qwen4ExpBatchedDecodeWorkspace & workspace,
                                       std::vector<std::vector<float>> & out_logits);

}  // namespace luce::common
