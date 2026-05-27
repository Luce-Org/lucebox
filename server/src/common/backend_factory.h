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
#include "placement/remote_draft_config.h"

#include <memory>
#include <string>

namespace dflash::common {

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
    RemoteDraftConfig remote_draft;

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
};

// ─── Factory function ───────────────────────────────────────────────────
// Inspects model_path GGUF metadata, constructs the correct backend, and
// calls init(). Returns nullptr on failure (diagnostic printed to stderr).
std::unique_ptr<ModelBackend> create_backend(const BackendArgs & args);

// Returns the detected architecture string without creating a backend.
// Useful for early dispatch (e.g. printing which backend will be used).
std::string detect_arch(const char * model_path);

bool arch_supports_remote_draft(const std::string & arch);
bool arch_supports_pflash_compression(const std::string & arch);

}  // namespace dflash::common
