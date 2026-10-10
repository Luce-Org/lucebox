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
#include "hidden_states.h"
#include "sampler.h"
#include "image_prompt.h"
#include "concurrency/seq_engine.h"
#include "placement/draft_residency.h"

namespace luce::common {

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

    // Image input. A backend that supports it names the text its chat template
    // turns into the image marker, and binds decoded images to a rendered prompt.
    virtual bool supports_images() const { return false; }
    virtual std::string image_placeholder() const { return {}; }
    virtual ImagePrepareStatus prepare_images(std::vector<int32_t> & tokens,
                                              std::vector<EncodedImage> images,
                                              uint64_t context_capacity,
                                              uint64_t output_reserve,
                                              ImagePromptHandle & payload,
                                              std::string & error) const {
        (void) tokens;
        (void) context_capacity;
        (void) output_reserve;
        if (!images.empty()) {
            error = "this backend does not support image input";
            return ImagePrepareStatus::invalid;
        }
        payload.reset();
        return ImagePrepareStatus::ok;
    }

    // Print the "[<arch>-daemon] ready ..." banner on stdout.
    virtual void print_ready_banner() const = 0;
    virtual int prefill_chunk_size() const { return 0; }
    // The context one sequence can hold, from a backend that can fit it to
    // its devices (DevicePlacement::fit_ctx); 0 from one that cannot.
    virtual int context_limit() const { return 0; }

    // ── Park / unpark ────────────────────────────────────────────────
    // Backend decides which resources to release/restore. Returns true on
    // success; on failure prints to stderr and returns false.
    virtual bool park(ParkTarget target) = 0;
    virtual bool unpark(ParkTarget target) = 0;
    virtual bool is_target_parked() const = 0;

    // ── Eviction (runtime model swapping) ────────────────────────────
    // evict() releases every device allocation the model owns (weights,
    // drafts, KV/recurrent caches, graphs, scratch and its backend contexts)
    // so another model can use the device. Host-side prefix snapshots stay
    // valid; reinstate() rebuilds the device layout that existed at init
    // (same context, KV types and sizes), so they restore afterwards.
    // Both run on the model's worker thread while it is idle. Unlike park(),
    // which a backend uses for its own transient memory juggling, eviction
    // always leaves the model with no device state.
    // cold: release everything. warm: keep the weights in device memory and
    // release the rest (KV/recurrent cache, graphs, scratch, backend context
    // with its pools), so a later reinstate only rebuilds the cache.
    // partial: release the primary device only, keeping a secondary device's
    // expert tier (DeepSeek V4 with in-process experts; others reject it).
    enum class EvictLevel { cold, warm, partial };
    enum class EvictStatus {
        ok,          // device state released (or already evicted)
        rejected,    // refused before any teardown; the model is unchanged
    };
    struct EvictResult {
        EvictStatus status = EvictStatus::rejected;
        std::string error;
        // Deferred snapshots that could not be copied to host and were freed;
        // their prefix-cache entries no longer restore.
        int snapshots_lost = 0;
    };
    // Whether this configuration supports eviction; `reason` explains a no.
    virtual bool supports_eviction(std::string & reason) const {
        reason = "this backend does not support eviction";
        return false;
    }
    virtual EvictResult evict(EvictLevel level = EvictLevel::cold) {
        (void) level;
        EvictResult r;
        supports_eviction(r.error);
        return r;
    }
    // On failure the model is left evicted with its partial device state
    // released, so a retry or another model can use the device.
    virtual bool reinstate(std::string & error) {
        supports_eviction(error);
        return false;
    }
    virtual bool is_evicted() const { return false; }
    // Evicted warm: the weights still occupy device memory.
    virtual bool weights_resident() const { return !is_evicted(); }
    // Device bytes of the weights (target + draft) while they are resident.
    virtual size_t weight_device_bytes() const { return 0; }

    // ── Generation ───────────────────────────────────────────────────
    // Run a full prefill + decode cycle. Backend owns the strategy
    // (autoregressive, speculative, DDTree, …).
    // Decode contract for every backend, including restore paths: create one
    // ThinkingBudget from req.budget_hook and the effective output cap. Share
    // it with the seed and all decode strategies. Apply accepted candidates
    // before history/KV/emission; never process prompts or replay tokens.
    // A speculative cut must repair KV even if emission cancels the request.
    // Prefer the shared AR/speculative loops when adding a model adapter.
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

    // Whether generate() (with force_ar_decode) and seq_engine() fill
    // per-token log-probabilities when sampler.wants_logprobs(). The server
    // rejects logprobs requests for backends that do not.
    virtual bool supports_logprobs() const { return false; }

    // ── Hidden states (/v1/hidden_states) ────────────────────────────
    // Whether this backend can report the residual stream after chosen
    // blocks for a prompt: through compute_hidden_states() when it serves
    // one request at a time, and through SeqEngine::admit_hidden_states()
    // when seq_engine() is non-null. The server rejects the request
    // otherwise.
    virtual bool supports_hidden_states() const { return false; }
    struct HiddenStateShape {
        int n_layers = 0;  // blocks a request may index
        int n_embd = 0;    // floats per reported vector
    };
    virtual HiddenStateShape hidden_state_shape() const { return {}; }
    // Prefill `prompt` from position zero -- never from a prefix snapshot --
    // and read `spec`'s layers. No token is decoded. When `snap_slot` >= 0
    // the state after the whole prompt is also saved into that snapshot
    // slot (an end-of-prefill save, which leaves the prefill itself
    // unchanged). Serves the single-request worker only.
    virtual bool compute_hidden_states(const std::vector<int32_t> & prompt,
                                       const HiddenStatesSpec & spec,
                                       int snap_slot,
                                       const DaemonIO & io,
                                       HiddenStates & out,
                                       std::string & error) {
        (void) prompt; (void) spec; (void) snap_slot; (void) io; (void) out;
        error = "this model backend does not report hidden states";
        return false;
    }

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
    // Snapshots are right-sized to cur_pos and live in system RAM on
    // discrete GPUs. Hybrid models also copy their full recurrent state, so
    // each snapshot has a large fixed cost (see docs/PREFIX_CACHE.md).
    static constexpr int kMaxSlots = 64;

    // Memory held by the live cache and the saved snapshots, for /props.
    // Called on the thread that runs generation, never concurrently with it.
    struct MemoryReport {
        struct Cache {
            bool host = false;             // cache buffers live in host memory
            int64_t capacity_tokens = 0;   // positions one sequence can hold
            int64_t live_tokens = -1;      // committed positions; -1 when
                                           // paged or holding several
                                           // sequences
            uint64_t kv_bytes = 0;         // attention K/V
            uint64_t recurrent_bytes = 0;  // SSM/conv or compressor state
            uint64_t draft_feature_bytes = 0;  // DFlash target features held
                                               // in the cache allocation
            uint64_t other_bytes = 0;      // rollback, scratch rings, etc.
            uint64_t host_state_bytes = 0; // live host-side state beside the
                                           // cache (logits, feature windows)
        };
        struct Snapshot {
            int slot = -1;
            int tokens = 0;
            uint64_t bytes = 0;            // the snapshot buffer
            bool host = true;              // the buffer lives in host memory
            uint64_t host_copy_bytes = 0;  // host-side copies kept beside it
        };
        bool available = false;
        Cache cache;
        std::vector<Snapshot> snapshots;
    };
    virtual MemoryReport memory_report() const { return {}; }

    virtual bool snapshot_save(int slot) = 0;
    // snapshot_save() between generations, where the backend may keep the
    // live state as the snapshot and copy it out only before something
    // changes that state. A restore of `slot` as the very next operation
    // then continues the live state without any copy, and consumes it.
    // snapshot_ref() sees it only after snapshot_flush_deferred().
    virtual bool snapshot_save_deferred(int slot) { return snapshot_save(slot); }
    // Copy a snapshot snapshot_save_deferred() kept live into its slot.
    virtual void snapshot_flush_deferred() {}
    virtual void snapshot_free(int slot) = 0;
    virtual bool snapshot_used(int slot) const = 0;
    virtual int  snapshot_cur_pos(int slot) const = 0;

    // A snapshot lands at least this many positions past the restore point,
    // so a cut closer than that cannot be saved. Backends that start a
    // prefill chunk at every restore point (Qwen: beyond this distance) save
    // exactly there.
    virtual int snapshot_granularity() const { return 1; }

    // Minimum token distance between the prompt restore points a backend cuts
    // prefill at (spaced_restore_points). 0: every chat boundary. A backend
    // whose prompt numerics depend on where a forward ends cuts at every
    // restore point, so short messages would mean short, slow forwards.
    virtual int restore_point_spacing() const { return 0; }

    // True when the backend ends a prompt forward at every restore point
    // (GenerateRequest::restore_points). A tool-result request then keeps its
    // own last boundary among them, so a hit that skips the capture and a
    // miss that captures the tools head cut prefill alike.
    virtual bool prefill_cuts_at_restore_points() const { return false; }

    // System-memory bytes a snapshot of the first `tokens` positions would
    // take; positions beyond the cache capacity are clamped to it. Returns
    // 0 when the backend cannot estimate, which disables the prefix cache's
    // resident-memory budget for this backend.
    virtual size_t snapshot_bytes_estimate(int tokens) const {
        (void)tokens;
        return 0;
    }

    // Snapshot bytes the backend admits when it enforces its own allowance
    // (SIZE_MAX: no allowance). The prefix cache's automatic resident limit
    // stays within it, so the cache evicts before the backend refuses a capture.
    virtual size_t snapshot_allowance_bytes() const { return SIZE_MAX; }

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
        if (snapshot_used(slot)) {
            return merge_empty_spec_retry_result(
                result, restore_and_generate_impl(slot, retry, io));
        }
        // A deferred snapshot the first attempt continued in place no
        // longer exists (snapshot_save_deferred()); prefill the prompt.
        GenerateResult merged =
            merge_empty_spec_retry_result(result, generate_impl(retry, io));
        merged.restored_prefix_tokens = 0;
        return merged;
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
        // Exclusive end and width of the user-query token window inside
        // input_ids. Negative end preserves the legacy trailing-token window.
        // Keeping this separate from LUCE_COMPRESS_QUERY_TOKENS matters:
        // that knob controls lexical anchors, not neural scorer Q rows.
        int                  score_query_end = -1;
        int                  score_query_tokens = 8;
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

}  // namespace luce::common
