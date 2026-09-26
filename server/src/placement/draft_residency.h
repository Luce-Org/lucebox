// Drafter residency policy shared by draft-style runtime paths.
//
// The policy is intentionally scoped by draft use-case. PFlash compression can
// release its drafter immediately after prompt compression, while DFlash decode
// draft may need to stay resident across requests for latency.

#pragma once

#include <string>

namespace luce::common {

enum class DraftResidencyPolicy {
    Auto,
    Persistent,
    RequestScoped,
};

enum class DraftResidencyUse {
    PFlashCompress,
    DFlashDecode,
    MtpDecode,
};

enum class DraftResidencyAction {
    KeepLoaded,
    ReleaseAfterUse,
};

struct DraftResidencyContext {
    DraftResidencyUse use = DraftResidencyUse::PFlashCompress;
    bool low_vram_hint = false;
    bool has_decode_draft = false;
    // True when the startup skip-park probe proved the target, decode draft
    // and pflash drafter fit co-resident with margin. Under Auto it upgrades
    // PFlashCompress to KeepLoaded — the drafter stays loaded between
    // requests, removing its per-request reload. Ignored by other uses and
    // by explicit Persistent/RequestScoped policies.
    bool ample_vram = false;
};

inline const char * draft_residency_policy_name(DraftResidencyPolicy policy) {
    switch (policy) {
    case DraftResidencyPolicy::Auto:          return "auto";
    case DraftResidencyPolicy::Persistent:    return "persistent";
    case DraftResidencyPolicy::RequestScoped: return "request-scoped";
    }
    return "auto";
}

inline bool parse_draft_residency_policy(const std::string & value,
                                         DraftResidencyPolicy & out) {
    if (value == "auto") {
        out = DraftResidencyPolicy::Auto;
        return true;
    }
    if (value == "persistent") {
        out = DraftResidencyPolicy::Persistent;
        return true;
    }
    if (value == "request-scoped" || value == "request_scoped") {
        out = DraftResidencyPolicy::RequestScoped;
        return true;
    }
    return false;
}

inline DraftResidencyAction resolve_draft_residency_action(
        DraftResidencyPolicy policy,
        const DraftResidencyContext & ctx) {
    if (policy == DraftResidencyPolicy::Persistent) {
        return DraftResidencyAction::KeepLoaded;
    }
    if (policy == DraftResidencyPolicy::RequestScoped) {
        return DraftResidencyAction::ReleaseAfterUse;
    }

    switch (ctx.use) {
    case DraftResidencyUse::PFlashCompress:
        // Auto releases the pflash drafter after scoring on constrained
        // cards (resident drafter starves target prefill on 24GB; lazy
        // reload costs ~2s). With proven ample VRAM the same startup probe
        // that enabled skip-park also keeps the drafter resident — its
        // footprint is already accounted and the reload is pure overhead.
        return ctx.ample_vram
            ? DraftResidencyAction::KeepLoaded
            : DraftResidencyAction::ReleaseAfterUse;
    case DraftResidencyUse::DFlashDecode:
        // DFlash draft is latency-sensitive; keep it resident unless the
        // operator explicitly opted into the low-VRAM/request-scoped path.
        return (ctx.low_vram_hint && ctx.has_decode_draft)
            ? DraftResidencyAction::ReleaseAfterUse
            : DraftResidencyAction::KeepLoaded;
    case DraftResidencyUse::MtpDecode:
        // Placeholder use-case for future draft-style decode paths. Default to
        // persistent until a concrete MTP residency lifecycle is wired.
        return DraftResidencyAction::KeepLoaded;
    }
    return DraftResidencyAction::KeepLoaded;
}

}  // namespace luce::common
