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
#include <vector>

#include "ddtree.h"

struct ggml_tensor;
struct ggml_backend;

namespace dflash::common {

struct DFlashTarget {
    virtual ~DFlashTarget() = default;

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
                              std::vector<int32_t> * all_argmax = nullptr,
                              bool capture_ssm_intermediates = false) = 0;

    // Read the full [n_tokens x vocab] f32 logits produced by the most
    // recent verify_batch call. Used by sampled-verify (spec decode with
    // temperature). Returns false when the implementation does not keep
    // verify logits around.
    virtual bool read_verify_logits(int n_tokens, std::vector<float> & out) {
        (void)n_tokens; (void)out;
        return false;
    }

    // ── KV state management ─────────────────────────────────────────

    // Snapshot KV cache state before speculative verify, so it can be
    // rolled back if tokens are rejected.
    virtual bool snapshot_kv() = 0;

    // Restore KV cache to the last snapshot (undo speculative forward).
    virtual bool restore_kv() = 0;

    // Whether fast rollback is supported — uses per-step SSM intermediate
    // states captured during verify to restore recurrent state without replay.
    // When true, verify_batch captures intermediates and rollback_to() works.
    virtual bool supports_fast_rollback() const { return false; }

    // Roll back recurrent state to position `commit_n` within the last
    // verify batch (0-indexed). Uses SSM intermediate states captured during
    // verify. Also truncates KV to `base_pos + commit_n`. No replay needed.
    // Only valid when supports_fast_rollback() returns true.
    virtual bool rollback_to(int base_pos, int commit_n) {
        (void)base_pos; (void)commit_n; return false;
    }

    // ── DDTree tree-structured verify ───────────────────────────────
    // Whether this target can verify a draft tree (ancestor-masked batched
    // forward over DFS-ordered tree nodes). When false, callers fall back to
    // chain verify. Requires fast_rollback support (tree rollback reuses the
    // per-step SSM intermediate capture).
    virtual bool supports_tree_verify() const { return false; }

    // Verify a DDTree in one batched target forward. `flat_tokens` is the
    // DFS-ordered tree (slot 0 = root = previous round's bonus token, slots
    // 1..n_nodes = tree.token_ids), padded to `n_alloc` for graph reuse.
    // `committed` is the KV position of the root. On success fills
    // `posterior_out` (size 1+tree.n_nodes) with the target's argmax at each
    // tree node, and—when `logits_out` is non-null—the full [n_actual x vocab]
    // f32 logits (used to sample the bonus token at temperature > 0).
    virtual bool verify_tree(int committed,
                             const DDTree & tree,
                             const std::vector<int32_t> & flat_tokens,
                             int n_alloc,
                             std::vector<int32_t> & posterior_out,
                             std::vector<float> * logits_out = nullptr) {
        (void)committed; (void)tree; (void)flat_tokens; (void)n_alloc;
        (void)posterior_out; (void)logits_out;
        return false;
    }

    // Roll recurrent + KV state forward to the accepted tree path. `accepted_dfs`
    // is the list of DFS-tree indices of the committed nodes (from
    // follow_verified_tree, including the root at index 0). Restores SSM/conv
    // state to the deepest accepted node and compacts KV so the next round sees
    // a contiguous prefix. Only valid when supports_tree_verify() is true.
    virtual bool rollback_to_tree(int committed,
                                  const DDTree & tree,
                                  const std::vector<int> & accepted_dfs) {
        (void)committed; (void)tree; (void)accepted_dfs; return false;
    }

    // ── Token utilities ─────────────────────────────────────────────

    // Check if a token is end-of-sequence for this model.
    virtual bool is_eos(int token) const = 0;

    // Embed token IDs using the target's embedding table.
    // Output: `out` must have space for `n * hidden_size()` floats.
    // Optional GPU handles for the fused domino draft head. A target that
    // returns non-null for all three enables the single-graph draft-side path.
    virtual ggml_tensor *  lm_head_tensor()  { return nullptr; }
    virtual ggml_tensor *  gpu_embd_table()  { return nullptr; }
    virtual ggml_backend * fused_head_backend() { return nullptr; }

    virtual bool embed_tokens(const int32_t * tokens, int n,
                              float * out) const = 0;

    // ── LM head projection ──────────────────────────────────────────

    // Project draft hidden states through the target's lm_head
    // (out_norm + output weight) to get token IDs via argmax.
    // `hidden` has shape [n_tokens * hidden_size()].
    virtual bool project_hidden_to_tokens(const float * hidden,
                                          int n_tokens,
                                          std::vector<int32_t> & tokens_out) = 0;

    // Project draft hidden states through the target lm_head and return full
    // f32 logits with vocab as the fastest-changing dimension.
    // Default false (unsupported); Domino-capable targets override.
    virtual bool project_hidden_to_logits(const float * hidden,
                                          int n_tokens,
                                          std::vector<float> & logits_out) {
        (void)hidden; (void)n_tokens; (void)logits_out;
        return false;
    }

    // Project draft hidden states through the target lm_head and return the
    // per-position top-K log-probabilities + token ids (for DDTree building).
    // Default false (unsupported); tree-verify targets override.
    virtual bool project_hidden_to_topk(const float * hidden,
                                        int n_tokens,
                                        int K,
                                        float temperature,
                                        std::vector<float> & top_log_probs,
                                        std::vector<int32_t> & top_token_ids) {
        (void)hidden; (void)n_tokens; (void)K; (void)temperature;
        (void)top_log_probs; (void)top_token_ids;
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
};

} // namespace dflash::common
