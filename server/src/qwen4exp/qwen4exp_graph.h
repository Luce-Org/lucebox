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
#include "common/sampler.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace luce::common {

// QSA indexer block pooling: [idim, nb*r] keys -> [idim, nb], block b = mean of tokens r*b .. r*b+r-1.
ggml_tensor * qwen4exp_pool_blocks(ggml_context * c, ggml_tensor * keys, int64_t r);

// K/V span of the stable T=1 decode graph at kv_len. The sequence's first span (`base`, set on first use) leaves at
// least one 256-token window past its kv_len; later spans grow from it in 512-token steps. These are the spans a
// token-by-token decode rebuilds with, as a function of kv_len alone, so a verify row attends over the same span
// (same attention numerics) as plain decode at that position.
int64_t qwen4exp_stable_kv_span(int64_t & base, int64_t max_ctx, int64_t kv_len);

// The longest context a stable QSA graph attends over: the runtime-count top-k's range.
constexpr int64_t kQwen4ExpStableQsaMaxCtx = 262144;

// A retained graph's K/V bucket for kv_len keys: whole 256-key blocks, at most max_ctx (and, for QSA rows,
// kQwen4ExpStableQsaMaxCtx).
int64_t qwen4exp_kv_bucket(int64_t max_ctx, int64_t kv_len, bool qsa);

// The multi-slot decode graph's row ceiling: RDNA3 MMID supports at most four batch-invariant decode rows.
constexpr int kQwen4ExpMaxBatchedSlots = 4;
// The rows per slot of a batched verify (qwen4exp_verify_batched): a pending token and three drafts. Two slots fill one
// batch-invariant product (8 rows).
constexpr int kQwen4ExpBatchedVerifyRows = 4;

// What a T=1 decode or verify forward reads back in place of its logits rows (n_vocab floats each: a 5-row verify
// is 5 MB, over PCIe when the target is a discrete card).
struct Qwen4ExpRowOutputs {
    int top_n = 0;                  // in: candidates per row (sample_logits_top_n), 0 for none
    bool on_device = false;         // out: the rows stayed on the device; the fields below are filled
    std::vector<int32_t> greedy;    // each row's argmax, the lowest id on ties (a host scan's pick); empty with top_n
    std::vector<float> top_vals;    // [T][top_n]: each row's top_n largest logits, in no order
    std::vector<int32_t> top_ids;   // [T][top_n]: their ids
};

struct Qwen4ExpForwardResult {
    bool ok = false;
    // With out_rows->on_device: the forward's logits rows, on the device until the next forward on this cache.
    ggml_tensor * logits = nullptr;
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
// One later rank's graph of qwen4exp_mtp_draft_slots: n_slots draft rows at `pos` (the cache stands in for each).
Qwen4ExpGraphMemory qwen4exp_mtp_rank_slots_graph_memory(ggml_backend_t backend, const Qwen4ExpWeights & w,
    Qwen4ExpCache & cache, int n_slots, int pos);
// The multi-slot decode graph of n_slots rows, each at `pos` (dense, or as QSA rows).
Qwen4ExpGraphMemory qwen4exp_batched_graph_memory(ggml_backend_t backend, const Qwen4ExpWeights & w,
    Qwen4ExpCache & cache, int n_slots, int pos, bool qsa);

// One independent sequence span for batch eligibility and per-slot solo fallback.
struct Qwen4ExpForwardSegment {
    Qwen4ExpCache * cache = nullptr;
    const int32_t * tokens = nullptr;
    int n_tokens = 0;
    int pos0 = 0;
};

// Pure decision for validated spans: at most kQwen4ExpMaxBatchedSlots one-token rows, each dense in the solo
// path, or (with qsa_rows: the backend serves stable QSA rows) a QSA decode.
// use_qsa is the cached QSA capability.
bool qwen4exp_can_batch(const Qwen4ExpWeights & w,
                       const Qwen4ExpForwardSegment * segments, int n_segments, bool use_qsa,
                       bool qsa_rows = false);

// Run the trunk. `tokens` has n_tokens entries, processed as one contiguous
// single-sequence span at positions [pos0, pos0 + n_tokens). On success the
// cache is advanced to pos0 + n_tokens and out_logits holds n_vocab floats
// for the final token. out_hidden, when set, receives every token's final HC
// residual (n_embd * n_hc floats each) for the MTP draft head.
//
// verify (MTP speculation, 2 <= n_tokens <= k+1, qwen4exp_verify_supported): each
// token is computed exactly as a T=1 forward at its position computes it
// (batch-invariant matmuls, per-token attention), out_logits holds all rows,
// and the cache keeps what qwen4exp_verify_rollback needs to retain any accepted
// prefix (the recurrent state as it was plus each token's replay log, and the
// conv/PLE history after every token).
// mtp_prefill (n_tokens > 1, contiguous prompt chunks): append K/V-only MTP
// slices to this graph. out_hidden receives only the last pending trunk row.
// out_rows (T=1 decode and verify): the rows stay on the device (result.logits)
// and out_logits is left empty; out_rows receives what Qwen4ExpRowOutputs lists
// instead. Other forwards fill out_logits and leave out_rows->on_device false.
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
                                       const Qwen4ExpInputs * inputs = nullptr,
                                       Qwen4ExpRowOutputs * out_rows = nullptr);

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
// qwen4exp_mtp_draft for several slots (slot s: tokens[s], hidden[s], n[s] pending pairs from pos0[s]): each slot's
// first rank on its own, then every later rank once for all slots (the draft layer's weights read once for a row per
// slot, each slot attending over its own draft K/V), and one wait for every slot's k drafts. ranks[r] keeps rank r's
// retained graph (k entries). Drafts only propose; the verify decides every token.
bool qwen4exp_mtp_draft_slots(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache * const * caches,
        const int32_t * const * tokens, const float * const * hidden, const int * n, const int * pos0, int n_slots,
        int k, std::vector<Qwen4ExpBatchedDecodeWorkspace> & ranks, std::vector<std::vector<int32_t>> & drafts);

// Run all but `keep` pending pairs through the draft layer (K/V only), at most max_rows per forward.
bool qwen4exp_mtp_catch_up(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache,
                           Qwen4ExpMtpPending & pending, size_t keep,
                           int max_rows = std::numeric_limits<int>::max());

// How qwen4exp_mtp_step hands the verified logits rows to sample_row (Qwen4ExpRowOutputs).
enum class Qwen4ExpRows {
    all,      // every row read back after the forward; `logits` holds them all
    lazy,     // each row read when sample_row asks for it (none past a rejection), after its top candidates when the
              // step's top_n asks for them; `logits` ends with the step's last emitted row once generation ends
    greedy,   // the caller samples each row's argmax: sample_row gets the device's pick (the lowest id on ties) in its
              // sample; `logits` receives the last emitted row only in the step that ends generation (sample_row
              // returned false)
};

// The rows a request's verify reads: the device's picks for an argmax sampler, lazily for any other, every row
// with LUCE_GPU_ARGMAX=0 or when the caller checks against them (read_all).
inline Qwen4ExpRows qwen4exp_rows_for(const SamplerCfg & sampler, bool read_all = false) {
    if (read_all || !gpu_argmax_enabled()) return Qwen4ExpRows::all;
    return sample_logits_is_argmax(sampler) ? Qwen4ExpRows::greedy : Qwen4ExpRows::lazy;
}

// One verified row as sample_row sees it.
struct Qwen4ExpRow {
    bool picked = false;              // Qwen4ExpRows::greedy: the sample already holds the device's pick
    int top_n = 0;                    // > 0: the row's top_n largest logits and their ids, in no order
    const float * top_vals = nullptr;
    const int32_t * top_ids = nullptr;
    std::function<const float *()> read;   // the whole row (n_vocab floats), read from the device on first call

    const float * logits() const { return read(); }
};

struct Qwen4ExpMtpStep {
    int k = 0;                        // drafts verified
    Qwen4ExpMtpAcceptance decision;   // emitted[n_emitted - 1] is the next input
    bool more = true;                 // sample_row's last answer
    double draft_s = 0.0, verify_s = 0.0;
    double round_ms = -1.0;           // the whole step's time; -1 when it built a graph (not a steady-state sample)
    bool lookup = false;              // the verified drafts were the caller's lookup drafts, not the MTP's
    const char * error = nullptr;
};

// One decode step at `pos` feeding `token`: draft k tokens (the width controller's choice, or the cache's fixed
// draft count without one, capped at max_k), verify them with the token in one forward, sample each row with
// sample_row (fed token, the row, sample out; false ends generation) until a draft is rejected, roll the cache
// back to the retained rows and let the controller observe the step. `pending` (null: no draft layer) ends with
// `token` and, on success, with the next input. `logits` receives every verified row. on_drafts (test-only) may
// replace the drafts before verification. lookup_drafts (n_lookup > 0, with a draft layer) are verified instead of
// the MTP's: the draft layer does not run, its pending pairs stay and gain the retained ones, and the width
// controller does not observe the round (common/prompt_lookup.h prices it).
// `rows` says how the verified rows reach sample_row (Qwen4ExpRows); with Qwen4ExpRows::lazy, top_n > 0 hands each
// row's top_n candidates first.
bool qwen4exp_mtp_step(ggml_backend_t backend, const Qwen4ExpWeights & w, Qwen4ExpCache & cache, int pos,
                       int32_t token, int max_k, AdaptiveSpecWidth * width, Qwen4ExpMtpPending * pending,
                       const std::function<bool(int32_t, const Qwen4ExpRow &, int32_t &)> & sample_row,
                       std::vector<float> & logits, Qwen4ExpMtpStep & step,
                       const std::function<void(std::vector<int32_t> &)> & on_drafts = {},
                       const int32_t * lookup_drafts = nullptr, int n_lookup = 0,
                       Qwen4ExpRows rows = Qwen4ExpRows::all, int top_n = 0);

// Decode one next token for each independent slot. `caches[s]` owns that
// sequence's KV and recurrent state; `tokens[s]` and `positions[s]` are never
// interpreted as a common time axis. The shared workspace must outlive calls
// and is normally owned by the sequence-engine/model instance.
// A slot past the QSA boundary decodes as a stable T=1 QSA row inside the graph; if the
// backend lacks those rows or more than four slots are active, each slot runs its solo forward.
// Reference caches also use per-slot solo forwards. The server admits at most four slots.
// out_hidden, when set, receives slot `hidden_slot`'s final HC residual (n_embd * n_hc floats), as
// qwen4exp_forward's out_hidden would, so that slot's MTP head can draft once it decodes alone; with hidden_slot < 0,
// every slot's, slot-major.
// out_greedy, when set and the slots ran as one graph: every slot's argmax (the lowest id on ties, a host scan's
// pick), and out_logits holds only the rows logits_rows asks for (all when null; the others are left empty), so a
// greedy slot reads one id instead of its vocabulary row. Otherwise out_greedy is left empty and out_logits is full.
Qwen4ExpForwardResult qwen4exp_forward_batched(
                                       ggml_backend_t backend,
                                       const Qwen4ExpWeights & w,
                                       Qwen4ExpCache * const * caches,
                                       const int32_t * tokens,
                                       const int32_t * positions,
                                       int n_slots,
                                       Qwen4ExpBatchedDecodeWorkspace & workspace,
                                       std::vector<std::vector<float>> & out_logits,
                                       int hidden_slot = -1,
                                       std::vector<float> * out_hidden = nullptr,
                                       std::vector<int32_t> * out_greedy = nullptr,
                                       const uint8_t * logits_rows = nullptr);

// Allocation plan of the batched verify graph: n_slots slots of `rows` rows at `pos` (the cache stands in for each),
// dense or QSA as a T=1 decode at each row's position is.
Qwen4ExpGraphMemory qwen4exp_batched_verify_graph_memory(ggml_backend_t backend, const Qwen4ExpWeights & w,
    Qwen4ExpCache & cache, int n_slots, int rows, int pos);
// Every decoding slot's MTP verify in one graph: slot s feeds `rows` tokens (its pending token, then rows - 1 drafts),
// slot-major in `tokens`, at positions[s] onward. Each row computes what a T=1 decode at its position computes (the
// slot groups' batch-invariant products, each row's own T=1 attention, each slot's recurrence over its rows), and every
// slot keeps what its rollback needs as a solo verify does (qwen4exp_verify_rollback). A row past the QSA boundary
// attends as a stable verify graph's QSA row; slots and rows of both kinds mix freely. Needs every cache with the MTP
// rollback state (qwen4exp_verify_supported) and no KVFlash pool, and for QSA rows a backend that serves them in a
// retained graph (as a multi-slot decode step needs). False: not run (the caller decodes the step plainly). `out`
// (Qwen4ExpRowOutputs, top_n in) receives each row's argmax and, with top_n > 0, its top_n candidates; the logits rows
// stay on the device until the next call (qwen4exp_verify_batched_row). hidden: each row's final HC residual
// (n_embd * n_hc floats), slot-major.
bool qwen4exp_verify_batched_supported(ggml_backend_t backend, const Qwen4ExpWeights & w,
                                       Qwen4ExpCache * const * caches, const int32_t * positions, int n_slots,
                                       int rows);
Qwen4ExpForwardResult qwen4exp_verify_batched(ggml_backend_t backend, const Qwen4ExpWeights & w,
        Qwen4ExpCache * const * caches, const int32_t * tokens, const int32_t * positions, int n_slots, int rows,
        Qwen4ExpBatchedDecodeWorkspace & workspace, Qwen4ExpRowOutputs & out, std::vector<float> & hidden);
// Row `row` (slot-major) of the last qwen4exp_verify_batched's logits, n_vocab floats, into `out`.
bool qwen4exp_verify_batched_row(const Qwen4ExpBatchedDecodeWorkspace & workspace, const Qwen4ExpWeights & w, int row,
                                 std::vector<float> & out);

}  // namespace luce::common
