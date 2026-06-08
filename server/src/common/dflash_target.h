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

#include <cstdint>
#include <utility>
#include <vector>

namespace dflash::common {

// Per-position top-K distribution: [position][k] = (token_id, logit). Used by
// stochastic (Leviathan) acceptance at temp>0. Populated by verify_batch /
// project_hidden_to_tokens ONLY when stochastic capture is enabled.
using TopKDist = std::vector<std::vector<std::pair<int32_t, float>>>;

struct DFlashTarget {
    virtual ~DFlashTarget() = default;

    // ── Stochastic-acceptance distribution capture (default: unsupported) ──
    // When enabled, verify_batch/project_hidden_to_tokens stash per-position
    // top-K logits so the spec loop can run Leviathan accept at temp>0.
    virtual void set_stochastic_capture(bool /*on*/) {}
    virtual const TopKDist & last_target_topk() const { static const TopKDist e; return e; }
    virtual const TopKDist & last_draft_topk()  const { static const TopKDist e; return e; }

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

    // ── KV state management ─────────────────────────────────────────

    // Snapshot KV cache state before speculative verify, so it can be
    // rolled back if tokens are rejected.
    virtual bool snapshot_kv() = 0;

    // Restore KV cache to the last snapshot (undo speculative forward).
    virtual bool restore_kv() = 0;

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

    // ── Configuration for draft model ───────────────────────────────

    // Target's hidden dimension (draft model must match).
    virtual int hidden_size() const = 0;

    // Mask token ID in the target's vocabulary (used for noise input).
    virtual int mask_token_id() const = 0;

    // Which target layers to capture intermediate activations from.
    // The draft model's fc layer expects exactly this many feature slices.
    virtual const std::vector<int> & capture_layer_ids() const = 0;
};

} // namespace dflash::common
