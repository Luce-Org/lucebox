// dflash_target.h — Interface that any target model must implement to support
// DFlash speculative decoding with the universal DFlash draft model.
//
// The DFlash draft model (z-lab/DFlashDraftModel) is a single generic Qwen3-style
// architecture that works with ANY target model. It cross-attends to intermediate
// features captured during the target's forward pass, and outputs hidden states
// in the target's representation space. The target's own lm_head then projects
// those hidden states to token IDs.
//
// A target backend implements this interface to opt into DFlash spec decode.

#pragma once

#include "ddtree.h"

#include <cstdint>
#include <vector>

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
struct ggml_tensor;

namespace dflash27b {

struct DFlashTarget {
    virtual ~DFlashTarget() = default;

    // Return the ggml backend used by this target's graph compute.  Default
    // returns nullptr; callers (e.g. Qwen3.6 MTP) that want to build CUDA
    // cgraphs against the same backend should check this and fall back if
    // it's null.
    virtual ggml_backend_t backend() const { return nullptr; }

    // Optional: return the LM-head weight tensor on the target's backend
    // (shape [n_embd, n_vocab], used by ggml_mul_mat).  When non-null, the
    // Qwen3.6 MTP step graph fuses `mul_mat(W, x_normed) -> argmax` into
    // its own cgraph, skipping a hidden -> host -> separate-cgraph round
    // trip per step.  Default returns nullptr so existing targets (CPU
    // stubs) keep the project_hidden_to_* fallback path.
    virtual ggml_tensor * lm_head_weight() const { return nullptr; }

    // Optional: causal attention window the target's full-attn blocks use
    // (kv_len - fa_window).  The MTP head uses the same window so it sees
    // the same active context.  0 means full causal context.
    virtual int fa_window() const { return 0; }

    // ── Target forward ──────────────────────────────────────────────

    // Run a batch of tokens through the target model.  Returns the argmax
    // of the last token in `last_tok`.  If `all_argmax` is non-null, fills
    // it with argmax for every position (used during spec-decode verify).
    //
    // During forward, the target MUST capture intermediate activations at
    // the layers specified by capture_layer_ids() and store them in the
    // draft's feature ring (how this happens is implementation-defined).
    virtual bool verify_batch(const std::vector<int32_t> & tokens,
                              int base_pos,
                              int & last_tok,
                              std::vector<int32_t> * all_argmax = nullptr) = 0;

    // Tree-structured verify: run a flat DFS-ordered DDTree through the
    // target with an ancestor-only attention mask.  `flat_tokens[0]` is the
    // tree root (= last accepted token), `flat_tokens[1..N-1]` are the
    // DFS-ordered tree nodes (mirroring DDTree::token_ids).  `tree.n_nodes`
    // must equal flat_tokens.size() - 1.  On success, `out_argmax` is the
    // target's argmax at each of the N tree positions (size == N) and
    // (if non-null) `out_logits` is the raw logits laid out as
    // [N × vocab] floats.  Returns false by default so existing targets
    // that haven't wired tree-verify can be detected by callers; concrete
    // targets override to plug in build_target_step_tree + the tree mask.
    virtual bool verify_tree(const std::vector<int32_t> & flat_tokens,
                             const DDTree & tree,
                             int base_pos,
                             std::vector<int32_t> & out_argmax,
                             std::vector<float> * out_logits = nullptr) {
        (void)flat_tokens; (void)tree; (void)base_pos;
        (void)out_argmax;  (void)out_logits;
        return false;
    }

    // ── KV state management ─────────────────────────────────────────

    // Snapshot KV cache state before speculative verify, so it can be
    // rolled back if tokens are rejected.
    virtual bool snapshot_kv() = 0;

    // Restore KV cache to the last snapshot (undo speculative forward).
    virtual bool restore_kv() = 0;

    // Rollback DeltaNet SSM/conv + full-attn KV to the accepted-path tail of
    // the most recent verify_tree() call.  accepted_dfs[0] must be 0 (root).
    // Returns false if unsupported; callers must treat false as fatal in
    // multi-iteration tree-spec loops (poisoned KV/SSM otherwise).
    virtual bool restore_kv_at_dfs(const std::vector<int> & accepted_dfs) {
        (void)accepted_dfs;
        return false;
    }

    // Roll back DeltaNet SSM/conv + full-attn KV to slot `accept_n` of the
    // most recent verify_batch chain.  Requires chain capture enabled.
    // Postcondition: cache cur_pos = base_pos + accept_n + 1.
    // Returns false if unsupported; chain runner falls back to snapshot+recommit.
    virtual bool restore_kv_at_chain(int accept_n) {
        (void)accept_n;
        return false;
    }

    // Enable per-position DeltaNet intermediate capture in verify_batch.
    // Off by default; unsafe when n_tokens > max_verify_tokens.
    virtual void enable_chain_capture(bool /*on*/) {}

    // Record linear-chain topology before verify_batch so restore_kv_at_chain()
    // can locate the rollback slot.  Must be called before each capturable iter.
    virtual void capture_topology_for_chain(int /*n_tokens*/, int /*base_pos*/) {}

    // ── Token utilities ─────────────────────────────────────────────

    // Check if a token is end-of-sequence for this model.
    virtual bool is_eos(int token) const = 0;

    // Embed token IDs using the target's embedding table.
    // Output: `out` must have space for `n * hidden_size()` floats.
    virtual bool embed_tokens(const int32_t * tokens, int n,
                              float * out) const = 0;

    // ── LM head projection ──────────────────────────────────────────

    // Project draft hidden states through the target's lm_head
    // (out_norm + output weight) to get token IDs via argmax.
    // `hidden` has shape [n_tokens * hidden_size()].
    virtual bool project_hidden_to_tokens(const float * hidden,
                                          int n_tokens,
                                          std::vector<int32_t> & tokens_out) = 0;

    // Optional: project draft hidden states through the target's lm_head
    // and return the full raw logits (n_tokens * vocab floats) on host.
    // Used by MTP drafters that need a top-K surface for DDTree (the
    // argmax path above hides the distribution). Default returns false so
    // existing targets compile unchanged; concrete targets that wire it
    // up resize `logits_out` to n_tokens * vocab and return true. The
    // `out_vocab` param reports the vocab dim back to the caller.
    virtual bool project_hidden_to_logits(const float * /*hidden*/,
                                          int /*n_tokens*/,
                                          std::vector<float> & /*logits_out*/,
                                          int & out_vocab) {
        out_vocab = 0;
        return false;
    }

    // ── Configuration for draft model ───────────────────────────────

    // Target's hidden dimension (draft model must match).
    virtual int hidden_size() const = 0;

    // Mask token ID in the target's vocabulary (used for noise input).
    virtual int mask_token_id() const = 0;

    // Which target layers to capture intermediate activations from.
    // The draft model's fc layer expects exactly this many feature slices.
    virtual const std::vector<int> & capture_layer_ids() const = 0;

    // Return the backbone's final post-norm hidden state for the last committed
    // token (hidden_size() floats, F32).  Populated by verify_batch.
    // Returns nullptr if not yet available (e.g. before first verify_batch).
    // Default implementation returns nullptr; Qwen35DFlashTarget overrides it.
    virtual const float * last_hidden() const { return nullptr; }

    // Return the full post-norm hidden sequence from the MOST RECENT
    // verify_batch call: n_tokens * hidden_size() floats, F32, laid out as
    // [token_0_hidden, token_1_hidden, ..., token_{n_tokens-1}_hidden].
    // *out_n_tokens is set to the number of tokens captured (matches the
    // n_tokens passed to verify_batch).  Default returns nullptr.
    virtual const float * last_hidden_seq(int * out_n_tokens) const {
        if (out_n_tokens) *out_n_tokens = 0;
        return nullptr;
    }

    // Return the post-norm hidden at an ABSOLUTE sequence position, if that
    // position is covered by the most recent verify_batch's hidden capture.
    // The Qwen3.6 MTP head needs h_{base_pos-1} for its input pair at each
    // chain step, which equals last_hidden() only on the first chain step
    // (right after prefill); subsequent steps need a hidden from earlier in
    // the most recent verify_batch chunk.  Returns nullptr if out of range.
    virtual const float * hidden_at_pos(int abs_pos) const {
        (void)abs_pos;
        return nullptr;
    }

    // Pre-final-output-norm variant of hidden_at_pos.  Mirrors llama.cpp
    // PR #22673's `t_h_pre_norm`.  The Qwen3.6 MTP head's hnorm normalises
    // h_prev internally; feeding it the post-output-norm tensor double-
    // normalises and compounds per-depth rejection on D>=2 chains.  Spec-
    // chain callers must prefer this accessor for the outer h_prev_0 seed
    // and fall back to hidden_at_pos() only if it returns nullptr (e.g.
    // adapters that do not yet capture the pre-norm sequence).  Default
    // returns nullptr; Qwen35DFlashTarget overrides it when hidden-seq
    // capture is enabled.
    virtual const float * hidden_at_pos_pre_norm(int abs_pos) const {
        (void)abs_pos;
        return nullptr;
    }

    // Enable per-position post-norm + pre-norm hidden capture during the
    // next verify_batch calls. Default no-op; Qwen35DFlashTarget overrides.
    virtual void enable_hidden_seq_capture(bool /*on*/) {}

    // FULL_SEQ during prefill (warm_head_kv reads per-position); LAST_ROW_ONLY
    // during decode-side chain verifies. Default no-op.
    enum class VerifyCaptureScope { FULL_SEQ, LAST_ROW_ONLY };
    virtual void set_hidden_capture_scope(VerifyCaptureScope /*scope*/) {}
};

} // namespace dflash27b
