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

#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "sampler.h"
#include "placement/draft_residency.h"

namespace dflash::common {

enum class ParkTarget {
    // NOTE: Empty preserves Qwen3's existing no-target park/unpark behavior.
    Empty,
    All,
    TargetModel,
    DraftModel,
};

struct ParkTargetMapping {
    ParkTarget target;
    std::string_view str_value;
};

inline constexpr ParkTargetMapping park_target_to_str_mappings[] = {
    {ParkTarget::Empty,       ""},
    {ParkTarget::All,         "all"},
    {ParkTarget::TargetModel, "target"},
    {ParkTarget::DraftModel,  "draft"},
};

constexpr const char * park_target_name(ParkTarget target) {
    for (const auto & mapping : park_target_to_str_mappings) {
        if (mapping.target == target) {
            return mapping.str_value.empty()
                ? "empty"
                : mapping.str_value.data();
        }
    }
    return "unknown";
}

constexpr std::optional<ParkTarget> parse_park_target(std::string_view value) {
    for (const auto & mapping : park_target_to_str_mappings) {
        if (mapping.str_value == value) return mapping.target;
    }
    return std::nullopt;
}

constexpr bool park_target_includes_target_model(ParkTarget target) {
    return target == ParkTarget::Empty ||
           target == ParkTarget::All ||
           target == ParkTarget::TargetModel;
}

constexpr bool park_target_includes_draft_model(ParkTarget target) {
    return target == ParkTarget::Empty ||
           target == ParkTarget::All ||
           target == ParkTarget::DraftModel;
}

// Token callback for streaming generation. Called once per committed token.
// Return true to continue generation, false to abort.
using TokenCallback = std::function<bool(int32_t token)>;

// Inference observer callback for live status updates. Called by backends
// at each spec-decode step to report phase/detail. When empty, backends
// skip the call (zero overhead).
//   phase: "draft", "verify", "accept", "prefill_chunk"
//   detail: JSON string with step-specific data
using InferenceObserver = std::function<void(const char * phase,
                                             const std::vector<int32_t> & tokens)>;

// ─── I/O handle passed to backend methods that need protocol output ─────
struct DaemonIO {
    int stream_fd = -1;

    // Optional token callback. When set, emit() calls this for each token
    // (excluding the -1 sentinel). If it returns false, the `cancelled`
    // flag is set and the caller should abort generation.
    TokenCallback on_token;
    mutable bool cancelled = false;

    // Optional inference observer for /status page. When set, backends call
    // this at each spec-decode step with draft tokens and phase info.
    InferenceObserver observer;

    // Write a single int32 to the stream fd (token or -1 sentinel).
    // Also invokes on_token if set. Sets cancelled=true if on_token
    // returns false (client disconnected).
    void emit(int32_t v) const;

    // Return an IO handle that also invokes `cb` for emitted tokens.
    DaemonIO with_token_callback(const TokenCallback & cb) const;
};

// ─── Generate request/result ────────────────────────────────────────────

// Thinking-budget force-close hook. Mirrors antirez/ds4 ds4_eval.c's
// hard_limit_reply_budget semantics: when the budget remaining (n_gen
// minus tokens committed so far) falls to hard_limit_remaining, the
// next sampled tokens get overridden with close_token_ids in order,
// giving the model the remaining budget to write a visible answer
// after the injected close-tag sequence.
//
// Single vs multi-token close:
//   Qwen3.6: </think> is one added_token (id 248069). close_token_ids
//            has size 1. One override + budget_close_injected=true.
//   DeepSeek/laguna: </think> tokenizes to 3 ordinary tokens
//            ([1718, 37947, 32] for DS-V3). close_token_ids has
//            size 3. Three consecutive overrides, then resume.
//
// This is "Level 2" of our thinking-budget migration: in-process
// mid-stream force-close, KV-continuous. Beats Level 1's phase-2
// reprompt because the model never sees a fresh prefill — its KV
// state continues naturally after the injected close.
//
// Current implementation: AR-decode only. When budget_hook is set,
// backends MAY route generation through their AR path (skipping spec
// decode) — the perf trade-off is acceptable since this only kicks in
// for thinking-enabled requests. Spec-decode integration is a follow-up.
struct BudgetHook {
    // Multi-token close sequence injected when `(n_gen - committed)`
    // drops to `hard_limit_remaining`. For Qwen3.x this is the
    // canonical "Considering the limited time..." summarize-and-stop
    // lead-in (tokenized at server startup); for non-qwen arches it's
    // a single close-tag token. Empty = hook disabled.
    std::vector<int32_t> close_token_ids;
    int                  hard_limit_remaining = 0;
};

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
    // Optional env-gated dflash stall recovery: when spec decode is about to
    // emit early EOS after an action preamble, inject a bare tool-call XML
    // prefix and continue in AR with KV state intact.
    const std::vector<int32_t> * stall_tool_prefix_tokens = nullptr;
    const std::vector<int32_t> * stall_action_suffix_tokens = nullptr;
    const std::vector<int32_t> * stall_skip_tokens = nullptr;
    // Optional thinking-budget hook — see BudgetHook docs above.
    BudgetHook                 budget_hook;
    // Common retry knob. Upper layers set this after a speculative decode
    // path returns success but emits no tokens, so each backend can route the
    // retry through its existing AR path without copying retry policy.
    bool                       force_ar_decode = false;
};

// Stable, backend-independent generation failure categories. Backends should
// use these for recurrent failures so callers do not need to understand
// architecture-specific strings. `generate_error_code()` is the daemon/API
// wire representation and must remain backward-compatible once published.
enum class GenerateErrorCode {
    Incomplete,
    AdapterUnavailable,
    ContextOverflow,
    SamplingUnsupported,
    PrefillFailed,
    DecodeSeedMissing,
    DecodeFailed,
    InvalidSnapshotSlot,
    ModelParked,
    BackendSpecific,
};

constexpr std::string_view generate_error_code(GenerateErrorCode error) {
    switch (error) {
    case GenerateErrorCode::Incomplete:          return "incomplete";
    case GenerateErrorCode::AdapterUnavailable:  return "adapter_unavailable";
    case GenerateErrorCode::ContextOverflow:     return "context_overflow";
    case GenerateErrorCode::SamplingUnsupported: return "sampling_unsupported";
    case GenerateErrorCode::PrefillFailed:       return "prefill_failed";
    case GenerateErrorCode::DecodeSeedMissing:   return "decode_seed_missing";
    case GenerateErrorCode::DecodeFailed:        return "decode_failed";
    case GenerateErrorCode::InvalidSnapshotSlot: return "invalid_snapshot_slot";
    case GenerateErrorCode::ModelParked:         return "model_parked";
    case GenerateErrorCode::BackendSpecific:     return "backend_specific";
    }
    return "unknown_error";
}

struct GenerateError {
    GenerateErrorCode code = GenerateErrorCode::Incomplete;
    std::string detail;
};

struct GenerateResult {
    // Default to an incomplete failure so a backend must explicitly call
    // succeed() before returning a successful result.
    std::optional<GenerateError> error = GenerateError{};
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
    // True when decode emitted only tokens that the API layer suppresses
    // (for example an immediate EOS/EOT). This is semantically equivalent
    // to zero output for clients and should take the same AR retry path as
    // an empty token vector.
    bool                       empty_visible_output = false;

    bool ok() const {
        return !error.has_value();
    }

    std::string_view error_code() const {
        return error ? generate_error_code(error->code) : std::string_view{};
    }

    std::string_view error_detail() const {
        return error ? std::string_view(error->detail) : std::string_view{};
    }

    void succeed() {
        error.reset();
    }

    void fail(GenerateErrorCode code, std::string detail = {}) {
        error = GenerateError{code, std::move(detail)};
    }
};

// ─── Backend interface ──────────────────────────────────────────────────
struct ModelBackend {
    virtual ~ModelBackend() = default;

    // Print the "[<arch>-daemon] ready ..." banner on stdout.
    virtual void print_ready_banner() const = 0;

    // ── Park / unpark ────────────────────────────────────────────────
    // Backend decides which resources to release/restore. Returns true on
    // success; on failure prints to stderr and returns false.
    virtual bool park(ParkTarget target) = 0;
    virtual bool unpark(ParkTarget target) = 0;
    virtual bool is_target_parked() const = 0;

    // ── Generation ───────────────────────────────────────────────────
    // Run a full prefill + decode cycle. Backend owns the strategy
    // (autoregressive, speculative, DDTree, …).
    GenerateResult generate(const GenerateRequest & req, const DaemonIO & io) {
        GenerateResult result = generate_impl(req, io);
        if (!should_retry_empty_spec_decode(req, result)) return result;

        std::fprintf(stderr,
            "[backend] spec-decode produced zero tokens after %.3f s decode; "
            "retrying with AR decode\n",
            result.decode_s);
        GenerateRequest retry = req;
        retry.force_ar_decode = true;
        return merge_empty_spec_retry_result(result, generate_impl(retry, io));
    }

    virtual GenerateResult generate_impl(const GenerateRequest & req,
                                         const DaemonIO & io) = 0;

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
    GenerateResult restore_and_generate(int slot, const GenerateRequest & req,
                                        const DaemonIO & io) {
        GenerateResult result = restore_and_generate_impl(slot, req, io);
        if (!should_retry_empty_spec_decode(req, result)) return result;

        std::fprintf(stderr,
            "[backend] restored spec-decode slot=%d produced zero tokens after "
            "%.3f s decode; retrying with AR decode\n",
            slot, result.decode_s);
        GenerateRequest retry = req;
        retry.force_ar_decode = true;
        return merge_empty_spec_retry_result(result,
                                             restore_and_generate_impl(slot, retry, io));
    }

    virtual GenerateResult restore_and_generate_impl(int slot,
                                                     const GenerateRequest & req,
                                                     const DaemonIO & io) = 0;

    static bool should_retry_empty_spec_decode(const GenerateRequest & req,
                                               const GenerateResult & result) {
        return req.n_gen > 0
            && !req.force_ar_decode
            && result.ok()
            && result.spec_decode_ran
            && (result.tokens.empty() || result.empty_visible_output);
    }

    static GenerateResult merge_empty_spec_retry_result(
            const GenerateResult & first, GenerateResult retry) {
        retry.prefill_s += first.prefill_s;
        retry.decode_s += first.decode_s;
        retry.accept_rate = first.accept_rate;
        retry.spec_decode_ran = first.spec_decode_ran || retry.spec_decode_ran;
        retry.budget_forced_close =
            first.budget_forced_close || retry.budget_forced_close;
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

    // Layer-split capability introspection. Non layer-split backends keep the
    // default false; LayerSplitBackend proxies model-adapter support.
    virtual bool supports_kvflash() const { return false; }
    virtual bool supports_mixed_backend_layer_split() const { return false; }

    // ── Routing data collection ──────────────────────────────────────
    // Set an external routing collector that the backend will call for each
    // token/layer during decode (hidden state + expert IDs). Used by
    // --collect-routing for predictor training data.
    //
    // Lifetime: the collector pointer is borrowed, not owned. The caller must
    // keep it alive until set_routing_collector(nullptr) is called (or the
    // backend is destroyed), and must not pass a collector to a backend that
    // decodes on another thread without outliving that decode.
    //
    // Returns true if the backend supports routing collection (MoE backends).
    // The default returns false so the server can detect unsupported backends
    // and warn instead of silently collecting nothing.
    virtual bool set_routing_collector(class MoeRoutingCollector *) { return false; }

    // Get the current routing stats (if tracked). Returns nullptr if the
    // backend does not support routing stats or they are not enabled.
    virtual const struct MoeHybridRoutingStats * get_routing_stats() const { return nullptr; }

    // ── Cleanup ──────────────────────────────────────────────────────
    // Release all resources (weights, cache, snapshots, drafter).
    // Called by run_daemon() before returning.
    // Spark day-one bootstrap: when true, the server feeds local agent history
    // (Claude Code + Codex) through generate() before serving, then calls
    // spark_bootstrap_finalize to save the profile and rebuild placement so the
    // first session is already calibrated. Default: unsupported (live-traffic
    // calibration still applies).
    virtual bool spark_wants_bootstrap() const { return false; }
    virtual bool spark_bootstrap_finalize(const std::string & profile_path) {
        (void)profile_path; return false;
    }

    virtual void shutdown() = 0;
};

}  // namespace dflash::common
