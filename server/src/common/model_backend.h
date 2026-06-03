// Model daemon backend interface.
//
// Abstract base class that encapsulates all model-specific operations so a
// single generic daemon loop (daemon_loop.cpp) can service any architecture
// (qwen35, laguna, qwen3, gemma, …) without duplicating the stdin/stdout
// protocol parsing.
//
// Concrete backends own their GPU resources, weight/cache lifecycle, and
// generation strategy (autoregressive, speculative decode, etc.).

#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "sampler.h"
#include "placement/draft_residency.h"

namespace dflash::common {

// Token callback for streaming generation. Called once per committed token.
// Return true to continue generation, false to abort.
using TokenCallback = std::function<bool(int32_t token)>;

// ─── I/O handle passed to backend methods that need protocol output ─────
struct DaemonIO {
    int stream_fd = -1;

    // Optional token callback. When set, emit() calls this for each token
    // (excluding the -1 sentinel). If it returns false, the `cancelled`
    // flag is set and the caller should abort generation.
    TokenCallback on_token;
    mutable bool cancelled = false;

    // Write a single int32 to the stream fd (token or -1 sentinel).
    // Also invokes on_token if set. Sets cancelled=true if on_token
    // returns false (client disconnected).
    void emit(int32_t v) const;

    // Return an IO handle that also invokes `cb` for emitted tokens.
    DaemonIO with_token_callback(const TokenCallback & cb) const;
};

// ─── Generate request/result ────────────────────────────────────────────

// Thinking-budget force-close hook; see docs/specs/thinking-budget.md.
// When (n_gen - committed) == hard_limit_remaining, overrides sampled
// tokens with close_token_ids (AR path only). Empty = disabled.
struct BudgetHook {
    // Inject sequence written when the hard cap fires OR when soft-close
    // fires. This is the verbatim tokenization of the model card's
    // `thinking_terminator_hint` (e.g. for Qwen3.6 the lead-in
    // "Considering the limited time by the user, ... </think>\n\n").
    // May be many tokens long; the first element is what the AR loop
    // writes on the firing step, with the rest streamed out on
    // subsequent steps. Empty = disabled.
    std::vector<int32_t> close_token_ids;
    // Short PROBE sequence used by the soft-close logit-ratio peek.
    // Conceptually this is the tokenization of just the close MARKER
    // (e.g. `</think>` — a single token id 248069 on Qwen3.6) rather
    // than the full inject directive above. Splitting probe-vs-inject
    // matters because the inject sequence for trained-hint models
    // starts with a content token like "Considering" whose logit is
    // 19-35 nats below the chosen token at every step, masking the
    // close-marker's true probability and preventing soft-close from
    // ever firing.
    // When empty, the soft-close peek falls back to
    // `close_token_ids.front()` (legacy behavior — kept so models that
    // haven't been updated keep working identically to before the split).
    std::vector<int32_t> soft_close_probe_ids;
    int                  hard_limit_remaining = 0;
    // Soft-close (Level 2 voluntary). When > 0, at each AR step the
    // loop compares the probe-token logit against the chosen-token
    // logit; if `prob[probe[0]] / prob[chosen] >= soft_close_min_ratio`
    // (equivalently `logit[probe[0]] - logit[chosen] >= log(min_ratio)`),
    // the inject sequence (close_token_ids) is written BEFORE the hard
    // limit is reached. 0.0 = disabled (default); 1.0 = fire only when
    // the probe token is already the most-likely token; lower values =
    // fire more aggressively. See docs/specs/thinking-budget.md §7 and
    // docs/experiments/soft-close-thinking-termination-plan.md.
    float                soft_close_min_ratio = 0.0f;
    // Minimum thinking tokens before soft-close is allowed to fire.
    // Soft-close peek runs on every AR step but the fire decision is
    // gated by this floor — protects against premature termination on
    // prompts where the close-marker logit briefly spikes mid-thought.
    // 0 = floor disabled (default). Per empirical trajectory data on
    // qwen3.6-27b (5 diverse prompts), </think> only becomes
    // argmax-competitive at 66-94% of natural reasoning length — so a
    // floor in the 64-256 range is the typical operating point.
    int                  soft_close_min_tokens = 0;
    // Diagnostic: when true, emit one stderr line per AR step inside the
    // thinking phase with (committed, chosen_tok, logit[probe0],
    // logit[chosen], diff). Used to record the close-vs-chosen logit
    // trajectory across a full thinking run so a sliding-threshold curve
    // can be designed from empirical data rather than guessed. Zero cost
    // when off. See server_main.cpp --debug-thinking-logits.
    bool                 debug_thinking_logits = false;

    // Probe token id used by the soft-close peek. Returns the first
    // element of soft_close_probe_ids when set, otherwise falls back to
    // close_token_ids.front() (legacy behavior). Callers must guard
    // against an empty hook before calling this.
    int32_t soft_close_probe_token() const {
        if (!soft_close_probe_ids.empty()) return soft_close_probe_ids.front();
        return close_token_ids.front();
    }
};

namespace soft_close {

// Returns true when the soft-close comparator would fire on this AR
// step. Side-effect free; safe to call from unit tests.
//
// Fast path: returns false in O(1) when min_ratio <= 0 (the disabled
// default). When the model has already chosen the close token on its
// own, also returns false — the natural-close path handles that.
//
// Math: `prob[i]/prob[j] = exp(logit[i] - logit[j])`, so
// `prob[close]/prob[chosen] >= min_ratio` ⟺
// `logit[close] - logit[chosen] >= log(min_ratio)`. We compare on
// logits to avoid `exp()` and full-softmax cost; this is numerically
// stable in fp32 for typical LLM logit ranges (~±20).
inline bool should_fire(const float * logits,
                        int32_t       chosen_tok,
                        int32_t       close0_tok,
                        float         min_ratio) {
    if (min_ratio <= 0.0f)          return false;
    if (chosen_tok == close0_tok)    return false;
    const float log_ratio = std::log(min_ratio);
    return (logits[close0_tok] - logits[chosen_tok]) >= log_ratio;
}

}  // namespace soft_close

struct GenerateRequest {
    std::vector<int32_t>       prompt;
    int                        n_gen       = 0;
    SamplerCfg                 sampler;
    bool                       do_sample   = false;
    bool                       stream      = false;  // emit tokens to stream_fd
    // Optional inline-snap: snapshot at this position after prefill.
    int                        snap_pos    = -1;
    int                        snap_slot   = -1;
    // Optional token callback for streaming. When set, backends call this
    // for each committed token. If it returns false, generation aborts
    // immediately. This is the primary mechanism for client-disconnect
    // cancellation in the native HTTP server.
    TokenCallback              on_token;
    // Tool call hint tokens: pre-tokenized structural tokens that are
    // predictable with ~100% confidence (XML tags, function name, param names).
    // When non-null, the spec decode loop uses these as draft overrides,
    // bypassing draft model computation for covered positions.
    const std::vector<int32_t> * hint_tokens = nullptr;
    // Optional thinking-budget hook — see BudgetHook docs above.
    BudgetHook                 budget_hook;
    // Per-request override for target spec-decode verify fa_window. Set by
    // http_server when pflash compresses, so verify sees the entire compressed
    // prompt (not just the last cfg_.fa_window positions). Zero = no override.
    int                        fa_window_override = 0;
    // Common retry knob. Upper layers set this after a speculative decode
    // path returns success but emits no tokens, so each backend can route the
    // retry through its existing AR path without copying retry policy.
    bool                       force_ar_decode = false;
};

struct GenerateResult {
    bool                       ok          = false;
    std::string                error;               // "prefill", "decode", etc.
    std::vector<int32_t>       tokens;
    double                     prefill_s   = 0.0;
    double                     decode_s    = 0.0;
    // True when the backend's Level 2 hook injected the </think> close
    // sequence during this generation (vs. the model self-closing). The
    // server uses this to attribute close_kind correctly: if the model
    // produced </think> naturally we report "natural"; if the hook fired
    // we report "hard". Without this flag, decoding the phase-1 token
    // stream and grepping for "</think>" cannot distinguish the two
    // (the injected close decodes identically).
    bool                       budget_forced_close = false;
    // True when the soft-close path (logit-ratio peek) injected the
    // </think> close sequence in this generation. Mutually exclusive
    // with budget_forced_close: when both could fire on the same step,
    // soft wins and budget_forced_close stays false. The server uses
    // this to attribute close_kind="soft" (vs "hard"). See
    // docs/specs/thinking-budget.md §7.
    bool                       soft_forced_close = false;
    // True iff the AR decode loop's post-close watchdog detected an n-gram
    // repetition loop and broke out early. Caller surfaces this so clients
    // can mark the answer as unreliable rather than treating the
    // (truncated) content as a clean response.
    bool                       degenerate_decode_close = false;
    // DFlash chain accept rate: accepted_draft_tokens / total_draft_positions.
    // 0.0 when spec decode did not run (AR fallback or no draft model).
    float                      accept_rate     = 0.0f;
    // True when spec decode actually ran (accept_rate==0 still needs a bandit update).
    bool                       spec_decode_ran = false;
};

// ─── Backend interface ──────────────────────────────────────────────────
struct ModelBackend {
    virtual ~ModelBackend() = default;

    // Print the "[<arch>-daemon] ready ..." banner on stdout.
    virtual void print_ready_banner() const = 0;

    // ── Park / unpark ────────────────────────────────────────────────
    // `what` is the tail of the command: "", "all", "target", "draft".
    // Backend decides which resources to release/restore. Returns true on
    // success; on failure prints to stderr and returns false.
    virtual bool park(const std::string & what) = 0;
    virtual bool unpark(const std::string & what) = 0;
    virtual bool is_target_parked() const = 0;

    // ── Generation ───────────────────────────────────────────────────
    // Run a full prefill + decode cycle. Backend owns the strategy
    // (autoregressive, speculative, DDTree, …).
    virtual GenerateResult generate(const GenerateRequest & req,
                                     const DaemonIO & io) = 0;

    GenerateResult generate_with_empty_spec_fallback(const GenerateRequest & req,
                                                     const DaemonIO & io) {
        GenerateResult result = generate(req, io);
        if (!should_retry_empty_spec_decode(req, result)) return result;

        std::fprintf(stderr,
            "[backend] spec-decode produced zero tokens; retrying with AR decode\n");
        GenerateRequest retry = req;
        retry.force_ar_decode = true;
        return merge_empty_spec_retry_result(result, generate(retry, io));
    }

    // ── Snapshots ────────────────────────────────────────────────────
    // With right-sized CPU-resident snapshots, each slot costs only
    // ~(cur_pos × 5 KB) of system RAM, so we can afford many slots.
    static constexpr int kMaxSlots = 64;

    virtual bool snapshot_save(int slot) = 0;
    virtual void snapshot_free(int slot) = 0;
    virtual bool snapshot_used(int slot) const = 0;
    virtual int  snapshot_cur_pos(int slot) const = 0;

    // RESTORE <slot> <prompt_path> <n_gen> — restore snapshot + generate.
    // Backend handles the diff-prefill and decode internally.
    virtual GenerateResult restore_and_generate(int slot,
                                                 const GenerateRequest & req,
                                                 const DaemonIO & io) = 0;

    GenerateResult restore_and_generate_with_empty_spec_fallback(
            int slot, const GenerateRequest & req, const DaemonIO & io) {
        GenerateResult result = restore_and_generate(slot, req, io);
        if (!should_retry_empty_spec_decode(req, result)) return result;

        std::fprintf(stderr,
            "[backend] restored spec-decode produced zero tokens; retrying with AR decode\n");
        GenerateRequest retry = req;
        retry.force_ar_decode = true;
        return merge_empty_spec_retry_result(result,
                                             restore_and_generate(slot, retry, io));
    }

    static bool should_retry_empty_spec_decode(const GenerateRequest & req,
                                               const GenerateResult & result) {
        return req.n_gen > 0
            && !req.force_ar_decode
            && result.ok
            && result.spec_decode_ran
            && result.tokens.empty();
    }

    static GenerateResult merge_empty_spec_retry_result(
            const GenerateResult & first, GenerateResult retry) {
        retry.prefill_s += first.prefill_s;
        retry.decode_s += first.decode_s;
        retry.accept_rate = first.accept_rate;
        retry.spec_decode_ran = first.spec_decode_ran || retry.spec_decode_ran;
        retry.budget_forced_close =
            first.budget_forced_close || retry.budget_forced_close;
        retry.soft_forced_close =
            first.soft_forced_close || retry.soft_forced_close;
        retry.degenerate_decode_close =
            first.degenerate_decode_close || retry.degenerate_decode_close;
        return retry;
    }

    // ── Snapshot serialization (for ondisk prefix cache) ─────────────
    // Read-only reference to a snapshot's ggml tensors for serialization.
    struct SnapshotRef {
        ggml_context        * ctx     = nullptr;
        ggml_backend_buffer_t buf     = nullptr;
        int                   cur_pos = 0;
        int32_t               last_tok = -1;  // last prefill token (for decode seeding)
    };

    // Export a snapshot's tensor context + buffer for read-only access.
    // Ownership is NOT transferred — caller must only read tensor data.
    // Returns empty ref (ctx==nullptr) if slot is invalid or unused.
    virtual SnapshotRef snapshot_ref(int slot) const { (void)slot; return {}; }

    // Import a deserialized snapshot into the given slot. Backend takes
    // ownership of ctx and buf on success. On failure (returns false),
    // the caller is responsible for freeing ctx and buf.
    virtual bool snapshot_adopt(int slot, ggml_context * ctx,
                                ggml_backend_buffer_t buf, int cur_pos,
                                int32_t last_tok = -1) {
        (void)slot; (void)ctx; (void)buf; (void)cur_pos; (void)last_tok;
        return false;
    }

    // ── Compress (pflash) ────────────────────────────────────────────
    // Backend owns the DrafterContext lifecycle and park/unpark policy.

    struct CompressRequest {
        std::vector<int32_t> input_ids;      // drafter-tokenized prompt
        float                keep_ratio;      // fraction to keep (0.0–1.0)
        std::string          drafter_path;    // GGUF path (for lazy-load)
        int                  drafter_gpu = 0;  // backend-local GPU for PFlash drafter
        bool                 skip_park = false; // true on >=32GB GPUs
        // Per-request transitive-cascade override (-1 = use env default).
        // 0 = off (agentic path: suppress cascade to avoid anchor bloat).
        // 1 = on  (retrieval path: full expansion, same as today).
        int                  use_transitive = -1;
        DraftResidencyAction residency_action = DraftResidencyAction::KeepLoaded;
    };

    struct CompressResult {
        bool                 ok = false;
        std::vector<int32_t> compressed_ids;  // surviving token IDs
    };

    // Typed compress API (preferred for in-process callers).
    virtual CompressResult compress(const CompressRequest & req);

    // Legacy string-based compress (for daemon_loop stdin protocol).
    // `line` is the full "compress ..." command line.
    virtual bool handle_compress(const std::string & line,
                                  const DaemonIO & io) = 0;
    virtual void free_drafter() = 0;

    // ── Arch-specific command hook ───────────────────────────────────
    // Called for any command the generic loop does not recognize. Return
    // true if the backend handled it; false to fall through to the
    // "unknown command" error path.
    virtual bool try_handle_command(const std::string & line,
                                     const DaemonIO & io) {
        (void)line; (void)io;
        return false;
    }

    // ── DFlash speculative decode support ────────────────────────────
    // Returns true if this backend can participate in DFlash spec decode
    // (i.e. it implements the DFlashTarget interface).
    virtual bool supports_dflash_spec_decode() const { return false; }

    // Return the DFlashTarget adapter for this backend. Only valid when
    // supports_dflash_spec_decode() returns true. Default returns nullptr.
    virtual class DFlashTarget * dflash_target() { return nullptr; }

    // Release oversized scratch buffers between requests to prevent VRAM
    // growth over time. Default is a no-op.
    virtual void release_scratch() {}

    // Return true when the backend can route draft execution through the
    // common remote-draft IPC transport. Model families that do not implement
    // the DFlash feature boundary keep the default false and are rejected by
    // the server before startup.
    virtual bool supports_remote_draft() const { return false; }

    // ── Cleanup ──────────────────────────────────────────────────────
    // Release all resources (weights, cache, snapshots, drafter).
    // Called by run_daemon() before returning.
    virtual void shutdown() = 0;
};

}  // namespace dflash::common
