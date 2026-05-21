// Backend factory — arch-detecting ModelBackend construction.
//
// Given a GGUF model path and placement options, inspects the file's
// `general.architecture` key and constructs the appropriate ModelBackend
// subclass (Qwen35Backend, LagunaBackend, Qwen3Backend, Gemma4Backend).
//
// This decouples backend creation from the daemon binary's argv parsing
// and allows both the daemon (test_dflash) and the new native server to
// share the same construction logic.

#pragma once

#include "model_backend.h"
#include "placement/placement_config.h"

#include <memory>
#include <string>

namespace dflash::common {

// ─── MTP source selection ────────────────────────────────────────────────
// Replaces the old free-form mtp_draft_source string (@howard0su #237, line 59).
enum class MtpSource {
    None,             // no MTP speculator
    Native,           // MTP heads co-located in the target GGUF (e.g. unsloth single-file)
    ExternalDrafter,  // separate MTP-head GGUF supplied via mtp_gguf_path
    Auto,             // probe target GGUF for nextn_predict_layers; Native if found, else None
};

// ─── Backend creation arguments ─────────────────────────────────────────
// A superset of all per-arch config fields. The factory reads only those
// relevant to the detected arch; unused fields are silently ignored.
struct BackendArgs {
    // Required
    const char *    model_path   = nullptr;   // target .gguf

    // Optional: speculative decode draft model (qwen35 only)
    const char *    draft_path   = nullptr;

    // Device placement
    DevicePlacement device;
    DevicePlacement draft_device;

    // I/O — only used when running under daemon_loop (legacy). The new
    // server passes -1 and uses on_token callbacks instead.
    int             stream_fd    = -1;

    // Chunked prefill
    int             chunk        = 512;

    // qwen35-specific speculative decode options
    int             fa_window        = 2048;
    int             kq_stride_pad    = 32;
    int             draft_swa_window = 0;
    int             draft_ctx_max    = 4096;
    bool            fast_rollback    = false;
    bool            seq_verify       = false;
    bool            ddtree_mode      = false;
    int             ddtree_budget    = 64;
    float           ddtree_temp      = 1.0f;
    bool            ddtree_chain_seed = true;
    bool            use_feature_mirror = false;

    // MTP (Multi-Token Prediction) speculator — mutually exclusive with --draft.
    // mtp_source drives which loading path is taken:
    //   None           → MTP disabled; mtp_gguf_path ignored.
    //   Native         → MTP heads embedded in model_path GGUF (single-file, e.g. unsloth).
    //                    mtp_gguf_path is left nullptr; the factory sets it to model_path.
    //   ExternalDrafter→ Separate MTP-head GGUF at mtp_gguf_path (required).
    //   Auto           → factory calls gguf_contains_mtp_tensors(model_path): if true,
    //                    resolves to Native; otherwise resolves to None.
    MtpSource    mtp_source       = MtpSource::None;
    const char * mtp_gguf_path    = nullptr;  // required only for ExternalDrafter
    int          mtp_gamma        = 0;        // 0 = MTP loaded but not active; >0 = chain depth
    bool         mtp_use_topk     = false;    // false = chain (default), true = mtp_topk strategy
    int          mtp_draft_topk   = 1;
};

// ─── Factory function ───────────────────────────────────────────────────
// Inspects model_path GGUF metadata, constructs the correct backend, and
// calls init(). Returns nullptr on failure (diagnostic printed to stderr).
// When args.mtp_source == Auto, resolves to Native or None before
// constructing; the resolved value is not written back into args.
std::unique_ptr<ModelBackend> create_backend(const BackendArgs & args);

// Returns the detected architecture string without creating a backend.
// Useful for early dispatch (e.g. printing which backend will be used).
std::string detect_arch(const char * model_path);

// Returns true if the GGUF at `path` contains MTP-head tensors.
// Heuristic: presence of `qwen35.nextn_predict_layers` metadata key with
// a value > 0. Pure metadata scan — no tensor allocation, no GPU touch.
// Used by create_backend() when mtp_source == Auto.
bool gguf_contains_mtp_tensors(const std::string & path);

}  // namespace dflash::common
