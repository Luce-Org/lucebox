// mtp_interface.h — Generic MTP (Multi-Token Prediction) module interface.
//
// Hosts multiple MTP designs under one outer abstraction:
//
//   - ExternalDrafter (Gemma4):  separate drafter weights with explicit
//     h_prev chain and cross-attention into the target's KV donor layers.
//   - NativeHeads (Qwen3.6):     MTP heads built into the backbone with
//     embedding + LM-head weight sharing; no explicit chain.
//
// A common base IMtpModule expresses what every MTP truly has; per-flavor
// mixins (IExternalDrafterMtp, INativeMtp) carry the flavor-specific
// surface. The γ-loop + verify + accept/rollback live once in
// MtpChainRunner (see mtp_chain_runner.h) and dispatch on `flavor()`.
//
// This file is peer to dflash_target.h and follows the same pattern:
// the target's existing DFlashTarget adapter provides everything the
// chain runner needs (verify_batch, snapshot_kv, restore_kv,
// embed_tokens, is_eos, hidden_size); MTP modules do not duplicate it.

#pragma once

#include <cstdint>
#include <vector>

namespace dflash27b {

struct DFlashTarget;  // forward — see common/dflash_target.h

namespace mtp {

// ── Flavor tag ──────────────────────────────────────────────────────────
// MtpChainRunner dispatches on this; concrete classes set it via the
// matching mixin (IExternalDrafterMtp / INativeMtp).
enum class MtpFlavor {
    ExternalDrafter,   // Gemma4-style: separate drafter, h_prev chain
    NativeHeads,       // Qwen3.6-style: MTP heads in the backbone
};

// ── Per-step value types ────────────────────────────────────────────────
//
// StepInput / StepOutput describe one γ-step of speculation. They are
// flavor-agnostic at the type level; flavor-specific fields are nullable
// (prev_hidden is consumed only by ExternalDrafter; NativeHeads ignores
// it and uses step_batch() to emit all γ tokens at once).

struct StepInput {
    int32_t       current_token   = -1;    // last accepted token id
    int           base_pos        = 0;     // committed target position
    int           gamma_index     = 0;     // 0..gamma-1 within the chain
    const float * prev_hidden     = nullptr;  // ExternalDrafter only; null otherwise
    int           prev_hidden_dim = 0;     // length of prev_hidden when non-null
};

struct StepOutput {
    int32_t              draft_token = -1;
    float                draft_logit = 0.0f;
    std::vector<float>   next_hidden;   // ExternalDrafter writes h_post; empty for NativeHeads

    // Optional top-K logprobs surface for tree-structured drafting (DDTree).
    // Empty when the module is configured for argmax-only drafting (the
    // default). When populated by NativeHeads with K>1, both vectors are
    // length K, sorted DESCENDING by logprob (rank 0 == argmax). For
    // multi-head emission the runner builds a [L * K] layout by stacking
    // the per-head vectors in order; each StepOutput holds the K entries
    // for its own depth.
    std::vector<float>   topk_logprobs;
    std::vector<int32_t> topk_ids;
};

// ── Common base ─────────────────────────────────────────────────────────
//
// Methods every MTP implementation truly has. LSP-safe: callers that
// only need flavor-agnostic lifecycle (attach / reset / shutdown) work
// against this base; flavor-specific entry points live on the mixins.

struct IMtpModule {
    virtual ~IMtpModule() = default;

    // Identifies which mixin (and therefore which entry point) this
    // module implements. Set via the matching mixin's `final` override;
    // do not override directly in concrete classes.
    virtual MtpFlavor flavor() const = 0;

    // Architectural ceiling on chain length. Qwen3.6 typically returns 2;
    // Gemma4 returns the drafter's trained chain depth.
    virtual int max_gamma() const = 0;

    // Requested operating γ for this module. Set once via set_effective_gamma
    // before the first decode. Orchestrator + chain runner read this — no
    // parallel storage anywhere else. Bug class blocked by construction:
    // gamma cannot disagree between caller and module.
    virtual int  effective_gamma() const = 0;
    virtual void set_effective_gamma(int gamma) = 0;

    // Backbone hidden size the module operates against. Must match the
    // target's DFlashTarget::hidden_size() exactly; chain runner asserts.
    virtual int hidden_size() const = 0;

    // Bind the module to its target (KV cache, embedding, EOS predicate,
    // LM-head projection). Called once before the first step; returns
    // false if shapes / arches are incompatible.
    virtual bool attach(DFlashTarget * target) = 0;

    // Clear any per-chain state (e.g. h_prev ring head, partial-accept
    // bookkeeping). Called by the runner between user requests.
    virtual void reset_chain() = 0;

    // Release all device + host resources. Called at backend shutdown.
    virtual void shutdown() = 0;

    // Seed h_prev for the first chain step (last post-norm hidden from prefill).
    // Default no-op; both ExternalDrafter and NativeHeads override.
    virtual void set_initial_hidden(const float * /*h_prev*/, int /*dim*/) {}
};

// ── ExternalDrafter mixin ───────────────────────────────────────────────
//
// For MTP designs whose drafter is a separate model that reads the
// target's intermediate KV state and propagates an h_prev hidden through
// the γ chain (Gemma4 today; future external drafters plug in here).

struct IExternalDrafterMtp : IMtpModule {
    MtpFlavor flavor() const final { return MtpFlavor::ExternalDrafter; }

    // Single drafter step. The runner threads `prev_hidden` (the
    // captured h_prev from the previous step or from the target's
    // post-norm hidden after the last commit) into `StepInput`.
    // On return, `StepOutput::next_hidden` carries the drafter's h_post
    // for the next iteration.
    virtual bool step(const StepInput & in, StepOutput & out) = 0;

    // Which target layers the drafter cross-attends to. Resolved at
    // load time (e.g. Gemma4's resolve_mtp_donor_layers). The runner
    // hands these to the target's DFlashTarget::verify_batch path so
    // the target captures activations at exactly these layers.
    virtual const std::vector<int> & donor_layers() const = 0;

    // Configure the target-side h_prev capture buffer.
    //   batch_mode=false  : single-row capture (γ=1 path).
    //   batch_mode=true   : write all n_tokens rows during verify so
    //                       partial-accept γ>1 can pick the right row
    //                       host-side without a re-capture forward.
    // `gamma_max` sizes the batch buffer.
    virtual bool enable_target_hidden_capture(bool batch_mode, int gamma_max) = 0;

    // For γ>1 partial-accept: the row of the post-norm hidden tensor
    // to read into prev_hidden on the next step. Default sentinel -1
    // means "last row" (matches γ=1 contract).
    virtual void set_capture_row(int row) = 0;

    // Host-readable copy of the captured h_prev for the next step.
    // `out` must have space for `hidden_size()` floats; `dim` is the
    // caller's expected dim and is asserted to match.
    virtual bool consume_captured_hidden(float * out, int dim) = 0;
};

// ── NativeHeads mixin ───────────────────────────────────────────────────
//
// For MTP designs where the heads live inside the target's backbone and
// emit multiple draft tokens in one forward (Qwen3.6 today; DeepSeek-V3
// style would fit here too).

struct INativeMtp : IMtpModule {
    MtpFlavor flavor() const final { return MtpFlavor::NativeHeads; }

    // Number of draft tokens emitted per call to step_batch().
    // Bounded by max_gamma(); typically 1–2 for Qwen3.6.
    virtual int num_heads() const = 0;

    // Emit up to `num_heads()` draft tokens in a single backbone-aware
    // forward. The runner calls this once per chain (no h_prev threading);
    // `out` is sized by the implementation to num_heads().
    virtual bool step_batch(int32_t current_token,
                            int base_pos,
                            std::vector<StepOutput> & out) = 0;

    // Configure per-head top-K logprob emission. Default K=1 means argmax
    // only (StepOutput.topk_* stays empty — pre-existing behavior). With
    // K>1, step_batch additionally fills StepOutput.topk_logprobs and
    // topk_ids (length K, sorted DESCENDING) on every emitted head, which
    // the DDTree builder consumes. Concrete impls override; the default
    // no-op keeps fake/stub subclasses compatible with the existing ABI.
    virtual void set_draft_topk(int /*k*/) {}

    // Multi-step autoregressive chain draft. Concrete implementations chain
    // their own forward `chain_depth` times, feeding the head's own
    // post-shared_head_norm hidden as h_prev for the next iteration. `out`
    // is sized to the number of drafts actually emitted (≤ chain_depth);
    // each element is populated like step_batch (draft_token / draft_logit
    // and optionally topk_*).
    //
    // Default implementation: forward to step_batch and return ALL emitted
    // drafts unchanged.  This preserves the pre-Phase-A semantics for
    // multi-head native modules (where step_batch emits `num_heads` drafts
    // per call) — the runner sees the same draft count it did before, and
    // `chain_depth` is effectively ignored for any module that hasn't
    // overridden step_chain.  Overriders (Qwen36MtpModule today) treat
    // chain_depth as authoritative.
    virtual bool step_chain(int32_t current_token,
                            int base_pos,
                            int /*chain_depth*/,
                            std::vector<StepOutput> & out) {
        return step_batch(current_token, base_pos, out);
    }

    // Pre-warm head K/V over all prefill positions. `hiddens` is the backbone's
    // per-position post-norm sequence laid out [tok0_hidden, ..., tokN_hidden].
    virtual bool warm_head_kv(const int32_t * /*prompt*/, int /*n_prompt*/,
                              int32_t /*prefill_next*/, const float * /*hiddens*/) {
        return true;
    }

    virtual bool snapshot_head_kv(std::vector<std::vector<uint8_t>> & /*out_buf*/,
                                  std::vector<int> & /*out_pos*/) const {
        return false;
    }

    virtual bool restore_head_kv(const std::vector<std::vector<uint8_t>> & /*buf*/,
                                 const std::vector<int> & /*pos*/) {
        return false;
    }

    // Range-warm a contiguous slot window [start_slot..start_slot+n_chunk).
    // Used by partial-WARM restore: cached prefix supplies head_kv slots
    // [1..snap_pos] via restore_head_kv; this fills [snap_pos..prompt_len]
    // using the new request's delta hiddens + the snapshot's h_{snap_pos-1}.
    // `hiddens` has n_chunk rows of hidden_size floats; row i is h_{start_slot+i-1}.
    // `prefill_next` is the input for the trailing slot (where slot == n_prompt).
    virtual bool warm_head_kv_range(const int32_t * /*prompt_tokens*/, int /*n_prompt*/,
                                    int /*start_slot*/, int /*n_chunk*/,
                                    int32_t /*prefill_next*/, const float * /*hiddens*/) {
        return false;
    }
};

}  // namespace mtp
}  // namespace dflash27b
