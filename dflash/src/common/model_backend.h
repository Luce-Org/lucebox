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

#include "sampler.h"

namespace dflash27b {

// Token callback for streaming generation. Called once per committed token.
// Return true to continue generation, false to abort.
using TokenCallback = std::function<bool(int32_t token)>;

// Forward-declare MTP module so the optional mtp() accessor below
// doesn't pull mtp_interface.h into every backend.
namespace mtp { struct IMtpModule; }

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
};

struct GenerateResult {
    bool                       ok          = false;
    std::string                error;               // "prefill", "decode", etc.
    std::vector<int32_t>       tokens;
    double                     prefill_s   = 0.0;
    double                     decode_s    = 0.0;
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
    static constexpr int kMaxSlots = 8;

    virtual bool snapshot_save(int slot) = 0;
    virtual void snapshot_free(int slot) = 0;
    virtual bool snapshot_used(int slot) const = 0;
    virtual int  snapshot_cur_pos(int slot) const = 0;

    // RESTORE <slot> <prompt_path> <n_gen> — restore snapshot + generate.
    // Backend handles the diff-prefill and decode internally.
    virtual GenerateResult restore_and_generate(int slot,
                                                 const GenerateRequest & req,
                                                 const DaemonIO & io) = 0;

    // ── Compress (pflash) ────────────────────────────────────────────
    // Backend owns the DrafterContext lifecycle and park/unpark policy.

    struct CompressRequest {
        std::vector<int32_t> input_ids;      // drafter-tokenized prompt
        float                keep_ratio;      // fraction to keep (0.0–1.0)
        std::string          drafter_path;    // GGUF path (for lazy-load)
        bool                 skip_park;       // true on ≥32GB GPUs
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

    // ── MTP speculative decode support ───────────────────────────────
    // Returns true if this backend has an MTP module bolted on (Gemma4's
    // external drafter, Qwen3.6's native heads, or any future MTP impl).
    // The MTP path is orthogonal to DFlash: both may be supported by the
    // same backend (caller picks via request method).
    virtual bool supports_mtp() const { return false; }

    // Return the IMtpModule adapter for this backend. Only valid when
    // supports_mtp() returns true. Default returns nullptr.
    // Forward-declared to avoid a header dependency from model_backend.h
    // on mtp_interface.h — backends include both as needed.
    virtual mtp::IMtpModule * mtp() { return nullptr; }

    // ── Cleanup ──────────────────────────────────────────────────────
    // Release all resources (weights, cache, snapshots, drafter).
    // Called by run_daemon() before returning.
    virtual void shutdown() = 0;
};

}  // namespace dflash27b
