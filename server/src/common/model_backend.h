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
#include <functional>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "sampler.h"

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
    // Optional thinking-budget hook — see BudgetHook docs above.
    BudgetHook                 budget_hook;
    // Per-request override for target spec-decode verify fa_window. Set by
    // http_server when pflash compresses, so verify sees the entire compressed
    // prompt (not just the last cfg_.fa_window positions). Zero = no override.
    int                        fa_window_override = 0;
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
