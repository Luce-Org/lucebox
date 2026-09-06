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

#include <algorithm>
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
#include "generation_types.h"
#include "sampler.h"
#include "concurrency/seq_engine.h"
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

// Return true when an in-flight request should stop. Backends poll this at
// their existing prefill/decode cancellation boundaries so cancellation does
// not depend on filling the socket's send buffer first.
using CancellationProbe = std::function<bool()>;

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

    // Optional request-liveness probe. The native HTTP server uses this to
    // propagate a peer disconnect detected by the client thread into backend
    // prefill and decode loops.
    CancellationProbe should_cancel;

    // Optional inference observer for /status page. When set, backends call
    // this at each spec-decode step with draft tokens and phase info.
    InferenceObserver observer;

    // Write a single int32 to the stream fd (token or -1 sentinel).
    // Also invokes on_token if set. Sets cancelled=true if on_token
    // returns false (client disconnected).
    void emit(int32_t v) const;

    // Poll external cancellation and latch the result locally. `cancelled`
    // remains worker-thread-owned; the probe itself may read atomic state.
    bool is_cancelled() const {
        if (!cancelled && should_cancel && should_cancel()) {
            cancelled = true;
        }
        return cancelled;
    }

    // Return an IO handle that also invokes `cb` for emitted tokens.
    DaemonIO with_token_callback(const TokenCallback & cb) const;
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

    // ── Concurrent serving ───────────────────────────────────────────
    // Backends that can hold several live sequences at once and execute a
    // batched decode over paged KV expose them as decode slots through a
    // SeqEngine (common/concurrency/seq_engine.h). Any additional
    // per-sequence model state is an implementation detail of that engine.
    // nullptr — the
    // default — means this backend serves one request at a time and the
    // server drives it through generate().
    //
    // The engine is owned by the backend; the returned pointer is borrowed
    // and stays valid until shutdown().
    virtual SeqEngine * seq_engine() { return nullptr; }

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
        retry.restored_prefix_tokens = (std::max)(
            first.restored_prefix_tokens, retry.restored_prefix_tokens);
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

    // Compress several independent prompt spans under one backend residency
    // window. The default preserves existing behavior; backends that park
    // large target/draft weights can override this to park once for the whole
    // batch instead of once per span.
    virtual std::vector<CompressResult> compress_batch(
        const std::vector<CompressRequest> & requests) {
        std::vector<CompressResult> results;
        results.reserve(requests.size());
        for (const auto & request : requests) {
            results.push_back(compress(request));
        }
        return results;
    }

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
