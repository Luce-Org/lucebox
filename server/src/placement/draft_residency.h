// Drafter residency policy shared by draft-style runtime paths.
//
// The policy is intentionally scoped by draft use-case. PFlash compression can
// release its drafter immediately after prompt compression, while DFlash decode
// draft may need to stay resident across requests for latency.

#pragma once

#include <string>

namespace dflash::common {

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
        // In auto mode, only release the PFlash drafter when the operator gave
        // a low-VRAM hint. That preserves the existing fast resident path while
        // allowing small-card setups to make room for decode draft/target state.
        return ctx.low_vram_hint
            ? DraftResidencyAction::ReleaseAfterUse
            : DraftResidencyAction::KeepLoaded;
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

}  // namespace dflash::common
