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

#include "backend_args.h"
#include "gguf_inspect.h"
#include "model_capabilities.h"
#include "model_backend.h"

#include <memory>
#include <string>
#include <vector>

namespace dflash::common {

struct BackendPreparation;

// Runtime facts resolved once from BackendArgs and shared by server admission
// and backend construction. Fields are private so callers cannot substitute
// an architecture that disagrees with model_path.
class ResolvedBackendPlan {
public:
    const GgufModelInfo & model() const { return model_; }
    const std::string & arch() const { return model_.arch; }
    const std::string & model_path() const { return model_path_; }
    PlacementBackend target_backend() const { return target_backend_; }
    PlacementBackend compiled_backend() const { return compiled_backend_; }
    const BackendFeatureConfig & features() const { return features_; }
    MoeStoragePolicy moe_storage_policy() const { return moe_storage_.policy; }
    MoeStoragePolicySource moe_storage_source() const { return moe_storage_.source; }

private:
    std::string          model_path_;
    GgufModelInfo        model_;
    BackendFeatureConfig features_;
    MoeStoragePolicyResolution moe_storage_;
    PlacementBackend    target_backend_ = PlacementBackend::Auto;
    PlacementBackend    compiled_backend_ = PlacementBackend::Auto;

    friend BackendPreparation prepare_backend(
        const BackendArgs & args,
        const BackendFeatureConfig & features);
};

enum class BackendPreparationError {
    None,
    InvalidRequest,
    ModelInspection,
    FeatureCompatibility,
};

struct BackendPreparation {
    ResolvedBackendPlan plan;
    BackendPreparationError error = BackendPreparationError::None;
    std::string message;

    // Accepted-but-inert options for the resolved architecture and placement.
    // Populated only when ok(); the caller decides how to surface them.
    std::vector<std::string> warnings;

    bool ok() const { return error == BackendPreparationError::None; }
};

// Resolve model metadata and compiled placement, then apply all cross-feature
// compatibility policy. This is the server's fail-fast factory entry point:
// server_main forwards the raw request and only handles the categorized result.
BackendPreparation prepare_backend(
    const BackendArgs & args,
    const BackendFeatureConfig & features = {});

// ─── Factory function ───────────────────────────────────────────────────
// Inspects model_path GGUF metadata, constructs the correct backend, and
// calls init(). Returns nullptr on failure (diagnostic printed to stderr).
std::unique_ptr<ModelBackend> create_backend(const BackendArgs & args);

// Uses facts already resolved from the same BackendArgs. The factory verifies
// that the plan belongs to args.model_path before dispatching.
std::unique_ptr<ModelBackend> create_backend(
    const BackendArgs & args,
    const ResolvedBackendPlan & plan);

// Returns the detected architecture string without creating a backend.
// Useful for early dispatch (e.g. printing which backend will be used).
std::string detect_arch(const char * model_path);

}  // namespace dflash::common
